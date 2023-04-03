/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed_volume.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/random_val.cuh> // helpers to generate random values, directions
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/trainer.h>

#include <nanovdb/NanoVDB.h>

#include <filesystem/path.h>
#include <stb_image/stb_image.h>
#include <assert.h>

#include <fstream>

using namespace tcnn;

NGP_NAMESPACE_BEGIN

Testbed::NetworkDims Testbed::network_dims_volume2image() const {
	NetworkDims dims;
	dims.n_input = m_volume2image.mode == Testbed::EVolume2ImageMode::DefaultRay ? 6 : 4;
	dims.n_output = 4;
	dims.n_pos = 3;
	return dims;
}

// Sample a transfer function
// The transfer function is a 1D array of vec4 values, where each value is a color and opacity.
// The density value is a float in the range [0, 1], and the transfer function is sampled
// linearly between the two closest values.
__host__ __device__ inline vec4 sample_transfer_function(const vec4* transfer_function, const int transfer_function_size, float density) {
    if (density <= 0.0f) return transfer_function[0];
    if (density >= 1.0f) return transfer_function[transfer_function_size - 1];
    float idx = density * (transfer_function_size - 1);
    int idx0 = int(idx);
    int idx1 = idx0 >= transfer_function_size - 1 ? idx0 : idx0 + 1;

    float t = idx - idx0;
    return transfer_function[idx0] * (1.0f - t) + transfer_function[idx1] * t;
}

__device__ inline bool walk_to_next_event(default_rng_t &rng, const BoundingBox &aabb, vec3 &pos, const vec3 &dir, const uint8_t *bitgrid, float scale) {
    while (1) {
        float zeta1 = random_val(rng); // sample a free flight distance and go there!
        float dt = -std::log(1.0f - zeta1) * scale; // todo - for spatially varying majorant, we must check dt against the range over which the majorant is defined. we can turn this into an optical thickness accumulating loop...
        pos += dir*dt;
        if (!aabb.contains(pos)) return false; // escape to the mooon!
        uint32_t bitidx = tcnn::morton3D(int(pos.x*128.f+0.5f),int(pos.y*128.f+0.5f),int(pos.z*128.f+0.5f));
        if (bitidx<128*128*128 && bitgrid[bitidx>>3]&(1<<(bitidx&7))) break;
        // loop around and try again as we are in density=0 region!
    }
    return true;
}

static constexpr uint32_t MAX_TRAIN_VERTICES = 4; // record the first few real interactions and use as training data. uses a local array so cant be big.

static constexpr uint32_t SAMPLE_TRAINING_RAYS = 1;

__device__ vec3 get_random_point_on_aabb_surface(BoundingBox aabb, default_rng_t rng, int &selected_face) {
    // Randomly select the face of the bounding box to sample a point on
    // Sample a random float and cast to int to not be biased towards the lower values
    selected_face = int(random_val(rng) * 5.9999999999f);

    // Randomly sample a position on the face of the bounding box
    vec3 position = random_val_3d(rng) * aabb.diag() + aabb.min;
    switch (selected_face) {
        case 0: position.x = aabb.min.x; break;
        case 1: position.x = aabb.max.x; break;
        case 2: position.y = aabb.min.y; break;
        case 3: position.y = aabb.max.y; break;
        case 4: position.z = aabb.min.z; break;
        case 5: position.z = aabb.max.z; break;
    }

    return position;
}

__device__ float random_normal_distribution(default_rng_t rng, float mean, float std_dev) {
    float u1 = random_val(rng);
    float u2 = random_val(rng);
    float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
    return mean + std_dev * z0;
}

// Define a function to convert a point to spherical coordinates
__device__ inline vec2 to_spherical_coordinates(const vec3& point, const vec3& center)
{
    const vec3 direction = normalize(point - center);

    // Compute the azimuthal angle
    const float azimuth = atan2(direction.y, direction.x);

    // Compute the polar angle
    const float polar = acos(direction.z);

    // Return the spherical coordinates
    return vec2(azimuth, polar);
}

__device__ inline void convert_ray_to_spherical_coordinates(const vec3 ray_origin, const vec3 ray_direction, BoundingBox aabb, vec2 &sphere_coordinates_1, vec2 &sphere_coordinates_2) {
    // Offset the ray_origin by the size of the bounding box to ensure that the ray_origin is outside of the bounding box
    // First get the diagonal length of the bounding box, we can use the non-squared length as we only care about it being outside the bounding box
    const vec3 diag = aabb.diag();
    const float diag_length = diag.x*diag.x + diag.y*diag.y + diag.z*diag.z;
    const vec3 offset_ray_origin = ray_origin - ray_direction * diag_length;

    // Now intersect the ray with the bounding box
    const vec2 intersections = aabb.ray_intersect(offset_ray_origin, ray_direction);

    // Get the two intersection points
    const vec3 intersection_point_1 = offset_ray_origin + ray_direction * intersections.x;
    const vec3 intersection_point_2 = offset_ray_origin + ray_direction * intersections.y;

    // Store the center of the bounding box
    const vec3 center = aabb.center();

    // Convert the intersection points to spherical coordinates
    sphere_coordinates_1 = to_spherical_coordinates(intersection_point_1, center);
    sphere_coordinates_2 = to_spherical_coordinates(intersection_point_2, center);
}

__device__ inline void convert_spherical_coordinates_to_ray(const vec2 sphere_point_1, const vec2 sphere_point_2, BoundingBox aabb, vec3 &ray_origin, vec3 &ray_direction, bool forward_ray_origin = true) {
    // Get the center of the bounding box
    const vec3 center = aabb.center();

    // Convert the spherical coordinates to cartesian coordinates
    const vec3 point_1 = vec3(cos(sphere_point_1.x) * sin(sphere_point_1.y), sin(sphere_point_1.x) * sin(sphere_point_1.y), cos(sphere_point_1.y));
    const vec3 point_2 = vec3(cos(sphere_point_2.x) * sin(sphere_point_2.y), sin(sphere_point_2.x) * sin(sphere_point_2.y), cos(sphere_point_2.y));

    // Get the ray origin and direction
    ray_origin = center + point_1 * aabb.diag();
    ray_direction = normalize(point_2 - point_1);

    // If we are not forwarding the ray_origin, then we are done
    if (!forward_ray_origin) return;

    // Forward the ray_origin to the first intersection point
    ray_origin = ray_origin + ray_direction * aabb.ray_intersect(ray_origin, ray_direction).x;
}

__device__ inline vec4 raytrace_single_ray(
    vec3 starting_pos,
    vec3 ray_direction,
    vec4* __restrict__ transfer_function,
    int transfer_function_size,
    const uint8_t* bitgrid,
    vec3 world2index_offset,
    float world2index_scale,
    BoundingBox aabb,
    default_rng_t rng,
    float scale,
    nanovdb::DefaultReadAccessor<nanovdb::Tree<nanovdb::RootNode<nanovdb::InternalNode<nanovdb::InternalNode<nanovdb::LeafNode<float>, 4>, 5>>>::BuildType> acc) {
            // We go front to back, so we start with 100% opacity
        vec4 col = {0.0f, 0.0f, 0.0f, 1.0f};

        // Define a position with a tiny offset from our original position
        vec3 pos = starting_pos;// + (random_val_3d(rng) - vec3(0.5f)) * 0.000001f;

        for (int iter=0;iter<128;++iter) {
            vec3 nanovdbpos = pos*world2index_scale + world2index_offset;
            float density = acc.getValue({int(nanovdbpos.x+random_val(rng)), int(nanovdbpos.y+random_val(rng)), int(nanovdbpos.z+random_val(rng))});
            vec4 current_color = sample_transfer_function(transfer_function, transfer_function_size, density);

            // Composite the color
            current_color.a = -exp(-current_color.a) + 1.f;

            // premultiply the alpha
            current_color.rgb = current_color.rgb * current_color.a;

            // blend the color (front to back)
            col.rgb += col.a*current_color.rgb;
            col.a = (1.f - current_color.a) * col.a;

            if (!walk_to_next_event(rng, aabb, pos, ray_direction, bitgrid, scale))
                break;
        }

        // As we are doing front to back we now need to invert the alpha channel
        col.a = 1.f - col.a;

        return col;
}

__global__ void volume2image_generate_training_data_kernel(uint32_t n_elements,
    Testbed::Volume2ImageRayInformation* __restrict__ rays_out,
    vec4* __restrict__ colors_out,
    vec4* __restrict__ transfer_function,
    int transfer_function_size,
    const void* nanovdb,
    const uint8_t* bitgrid,
    vec3 world2index_offset,
    float world2index_scale,
    BoundingBox aabb,
    default_rng_t rng,
    float distance_scale,
    float global_majorant,
    Testbed::EVolume2ImageRaySampling sampling_method
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    rng.advance(idx*256);

    float scale = distance_scale / global_majorant;
    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(nanovdb);
    auto acc = grid->tree().getAccessor();

    // Make a decision which method we use for sampling
    // With a higher priority we sample a ray which starts at the surface of the volume and goes into the volume
    // With a lower priority we sample a ray which randomly starts inside the volume and goes into a random direction
    const float chance = random_val(rng);
    vec3 starting_pos = vec3(0.0f);
    vec3 dir = vec3(0.0f);

    switch (sampling_method) {
        case Testbed::EVolume2ImageRaySampling::Basic: {
            // Randomly sample a starting position for the ray within the bounding box
            starting_pos = random_val_3d(rng) * aabb.diag() + aabb.min;
            // Randomly sample a direction for the ray
            dir = normalize(random_dir(rng));

            break;
        }
        case Testbed::EVolume2ImageRaySampling::BasicFaceWeight: {
            // Sample a ray which starts at the surface of the volume and goes into the volume
            if (chance < 0.6f) {

                int face;
                starting_pos = get_random_point_on_aabb_surface(aabb, rng, face);

                // Randomly sample a direction for the ray
                dir = normalize(random_dir(rng));

                // Flip the direction of the ray if it is pointing away from the bounding box
                if (dot(dir, starting_pos - aabb.center()) < 0.0f) dir = -dir;

            } else {
                // Randomly sample a starting position for the ray within the bounding box
                starting_pos = random_val_3d(rng) * aabb.diag() + aabb.min;
                // Randomly sample a direction for the ray
                dir = normalize(random_dir(rng));
            }
            break;
        }
        case Testbed::EVolume2ImageRaySampling::Connections:
        {
            // Select two random points in the volume
            starting_pos = random_val_3d(rng) * aabb.diag() + aabb.min;
            vec3 pos2 = random_val_3d(rng) * aabb.diag() + aabb.min;

            // Calculate the direction from pos1 to pos2
            dir = normalize(pos2 - starting_pos);
            break;
        }

        case Testbed::EVolume2ImageRaySampling::ConnectionsFaceWeight: {
            // Sample a ray which starts at the surface of the volume and ends at the volume surface
            if (chance < 0.6f) {

                int start_face = 0;
                starting_pos = get_random_point_on_aabb_surface(aabb, rng, start_face);

                // Until we have a different face, keep sampling
                int end_face = 0;
                vec3 pos2 = vec3(0.0f);
                do {
                    pos2 = get_random_point_on_aabb_surface(aabb, rng, end_face);
                } while (start_face == end_face);

                // Calculate the direction from pos1 to pos2
                dir = normalize(pos2 - starting_pos);
            } else {
                // Select two random points in the volume
                starting_pos = random_val_3d(rng) * aabb.diag() + aabb.min;
                vec3 pos2 = random_val_3d(rng) * aabb.diag() + aabb.min;

                // Calculate the direction from pos1 to pos2
                dir = normalize(pos2 - starting_pos);
            }
            break;
        }
        case Testbed::EVolume2ImageRaySampling::Spherical: {

            // Sample a point on a 3d sphere and map the sphere onto a unit cube
            // For that sample a random direction, have it intersect with the bounding box and then select a random direction

            // Randomly sample a direction for the ray
            dir = normalize(random_dir(rng));

            // Have a temporary starting point at the center of the bounding box
            starting_pos = aabb.center();

            // Calculate the intersection of the ray with the bounding box
            auto box_intersection = aabb.ray_intersect(starting_pos, dir);

            // Have a random value which is biased towards 1
            float bias = random_val(rng);
            bias = bias*bias*bias;

            // Calculate the t value for the intersection point
            float t = min(box_intersection.x, 0.0f) * (1.0f - bias);

            // Calculate the starting position
            starting_pos = starting_pos + dir * (t + 1e-6f);

            // Now pick the actual random direction, by picking a random point towards which the ray will go
            vec3 sphere_position = aabb.min + random_val_3d(rng) * aabb.diag();

            // Calculate the direction from the starting position to the sphere position
            dir = normalize(sphere_position - starting_pos);

            break;
        }
        default: {
            // Throw an invalid assert error, if we get here, as then the sampling method is not implemented
            assert(false);
            break;
        }
    }

    // Assign our output ray information
    rays_out[idx].position = starting_pos;
    rays_out[idx].direction = dir;

    // Do the ray tracing to determine the color of the ray
    // We will do multiple iterations of the ray tracing to get a better estimate of the color

    // Have a final color which we will average over the iterations
    vec4 final_color = {0.0f, 0.0f, 0.0f, 0.0f};

    // For N iterations do the ray tracing with a slightly modified origin
    // This will help us get a better estimate of the color
    for (int sample=0; sample < SAMPLE_TRAINING_RAYS; ++sample) {
        // Add the color to our final color
        final_color += raytrace_single_ray(starting_pos, dir, transfer_function, transfer_function_size, bitgrid, world2index_offset, world2index_scale, aabb, rng, scale, acc);
    }

    // Average the final color
    final_color /= static_cast<float>(SAMPLE_TRAINING_RAYS);

    // Assign the final color
    colors_out[idx] = final_color;
}

__global__ void volume2image_generate_training_data_kernel_spherical(uint32_t n_elements,
    vec4* __restrict__ coordinates_out,
    vec4* __restrict__ colors_out,
    vec4* __restrict__ transfer_function,
    int transfer_function_size,
    const void* nanovdb,
    const uint8_t* bitgrid,
    vec3 world2index_offset,
    float world2index_scale,
    BoundingBox aabb,
    default_rng_t rng,
    float distance_scale,
    float global_majorant,
    Testbed::EVolume2ImageRaySampling sampling_method
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    rng.advance(idx*256);

    float scale = distance_scale / global_majorant;
    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(nanovdb);
    auto acc = grid->tree().getAccessor();

    // Randomly sample two points in spherical coordinates
    // The first point is the starting point of the ray
    // The second point end position of the ray
    vec2 sphere_pos_1 = vec2(random_val(rng) * 2.0f * 3.14159265358979323846f, random_val(rng) * 3.14159265358979323846f);
    vec2 sphere_pos_2 = vec2(random_val(rng) * 2.0f * 3.14159265358979323846f, random_val(rng) * 3.14159265358979323846f);

    // Get the ray from the sphere coordinates
    vec3 starting_pos = vec3(0.0f);
    vec3 dir = vec3(0.0f);
    convert_spherical_coordinates_to_ray(sphere_pos_1, sphere_pos_2, aabb, starting_pos, dir);

    // Do the ray tracing to determine the color of the ray
    // We will do multiple iterations of the ray tracing to get a better estimate of the color

    // Have a final color which we will average over the iterations
    vec4 final_color = {0.0f, 0.0f, 0.0f, 0.0f};

    // For N iterations do the ray tracing with a slightly modified origin
    // This will help us get a better estimate of the color
    for (int sample=0; sample < SAMPLE_TRAINING_RAYS; ++sample) {
        // Add the color to our final color
        final_color += raytrace_single_ray(starting_pos, dir, transfer_function, transfer_function_size, bitgrid, world2index_offset, world2index_scale, aabb, rng, scale, acc);
    }

    // Average the final color
    final_color /= static_cast<float>(SAMPLE_TRAINING_RAYS);

    // Assign the final color
    colors_out[idx] = final_color;
    // Store the sphereical coordinates
    coordinates_out[idx] = vec4(sphere_pos_1, sphere_pos_2);
}

void Testbed::train_volume2image(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {

    // Auxiliary matrices for training
    const uint32_t batch_size = (uint32_t)target_batch_size;
    const uint32_t n_elements = batch_size;

    float distance_scale = 1.f/std::max(m_volume.inv_distance_scale,0.01f);

    // Depending on the mode do a different training step
    if (m_volume2image.mode == Testbed::EVolume2ImageMode::DefaultRay) {
        const uint32_t n_output_dims = 4;
        const uint32_t n_input_dims = 6;

        m_volume2image.training.rays.enlarge(n_elements);
        m_volume2image.training.colors.enlarge(n_elements);

        // Run our training kernel
        linear_kernel(volume2image_generate_training_data_kernel, 0, stream, n_elements,
          m_volume2image.training.rays.data(),
          m_volume2image.training.colors.data(),
          m_volume.transfer_function.data(),
          m_volume.transfer_function.size(),
          m_volume.nanovdb_grid.data(),
          m_volume.bitgrid.data(),
          m_volume.world2index_offset,
          m_volume.world2index_scale,
          m_render_aabb,
          m_rng,
          distance_scale,
          m_volume.global_majorant,
          m_volume2image.ray_sampling
        );

        m_rng.advance(n_elements*256);

        GPUMatrix<float> training_batch_matrix((float*)(m_volume2image.training.rays.data()), n_input_dims, batch_size);
        GPUMatrix<float> training_target_matrix((float*)(m_volume2image.training.colors.data()), n_output_dims, batch_size);

        auto ctx = m_trainer->training_step(stream, training_batch_matrix, training_target_matrix);

        if (get_loss_scalar) {
            m_loss_scalar.update(m_trainer->loss(stream, *ctx));
        }
    } else {
        const uint32_t n_output_dims = 4;
        const uint32_t n_input_dims = 4;

        m_volume2image.training.spherical_positions.enlarge(n_elements);
        m_volume2image.training.colors.enlarge(n_elements);
        // Run our training kernel
        linear_kernel(volume2image_generate_training_data_kernel_spherical, 0, stream, n_elements,
          m_volume2image.training.spherical_positions.data(),
          m_volume2image.training.colors.data(),
          m_volume.transfer_function.data(),
          m_volume.transfer_function.size(),
          m_volume.nanovdb_grid.data(),
          m_volume.bitgrid.data(),
          m_volume.world2index_offset,
          m_volume.world2index_scale,
          m_render_aabb,
          m_rng,
          distance_scale,
          m_volume.global_majorant,
          m_volume2image.ray_sampling
        );

        m_rng.advance(n_elements*256);

        GPUMatrix<float> training_batch_matrix((float*)(m_volume2image.training.spherical_positions.data()), n_input_dims, batch_size);
        GPUMatrix<float> training_target_matrix((float*)(m_volume2image.training.colors.data()), n_output_dims, batch_size);

        auto ctx = m_trainer->training_step(stream, training_batch_matrix, training_target_matrix);

        if (get_loss_scalar) {
            m_loss_scalar.update(m_trainer->loss(stream, *ctx));
        }
    }



	m_training_step++;

}

__global__ void init_rays_volume(
	uint32_t sample_index,
	vec3* __restrict__ positions,
	Testbed::VolPayload* __restrict__ payloads,
    Testbed::Volume2ImageRayInformation* __restrict__ ray_information,
	uint32_t *pixel_counter,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox aabb,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	default_rng_t rng,
	const uint8_t *bitgrid,
	float distance_scale,
	float global_majorant,
    bool evaluate_rays_at_bounding_box
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= resolution.x || y >= resolution.y) {
		return;
	}
	uint32_t idx = x + resolution.x * y;
	rng.advance(idx<<8);
	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	Ray ray = pixel_to_ray(
		sample_index,
		{x, y},
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask
	);

	if (!ray.is_valid()) {
		depth_buffer[idx] = MAX_DEPTH();
		return;
	}

	ray.d = normalize(ray.d);
	auto box_intersection = aabb.ray_intersect(ray.o, ray.d);
	float t = max(box_intersection.x, 0.0f);
	ray.advance(t + 1e-6f);
	float scale = distance_scale / global_majorant;

	if (t >= box_intersection.y || (!evaluate_rays_at_bounding_box && !walk_to_next_event(rng, aabb, ray.o, ray.d, bitgrid, scale))) {
		frame_buffer[idx] = vec4(0.0f ,0.0f, 0.0f, 1.0f);
		depth_buffer[idx] = MAX_DEPTH();
	} else {
		uint32_t dstidx = atomicAdd(pixel_counter, 1);
        ray_information[dstidx].position = ray.o;
        ray_information[dstidx].direction = ray.d;
		positions[dstidx] = ray.o;
		payloads[dstidx] = {ray.d, vec4(0.f), idx};
		depth_buffer[idx] = dot(camera_matrix[2], ray.o - camera_matrix[3]);
	}
}

__global__ void volume2image_render_kernel_gt(
    uint32_t n_pixels,
    ivec2 resolution,
    default_rng_t rng,
    BoundingBox aabb,
    vec4* transfer_function,
    int transfer_function_size,
    const vec3* __restrict__ positions_in,
    const Testbed::VolPayload* __restrict__ payloads_in,
    const uint32_t *pixel_counter_in,
    const void *nanovdb,
    const uint8_t *bitgrid,
    float global_majorant,
    vec3 world2index_offset,
    float world2index_scale,
    float distance_scale,
    vec4* __restrict__ frame_buffer
) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n_pixels || idx >= pixel_counter_in[0])
        return;
    uint32_t pixidx = payloads_in[idx].pixidx;

    uint32_t y = pixidx / resolution.x;
    if (y >= resolution.y)
        return;

    vec3 pos = positions_in[idx];
    vec3 dir = payloads_in[idx].dir;
    rng.advance(pixidx<<8);
    const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(nanovdb);
    auto acc = grid->tree().getAccessor();

    // ye olde delta tracker
    float scale = distance_scale / global_majorant;

    frame_buffer[pixidx] = raytrace_single_ray(pos, dir, transfer_function, transfer_function_size, bitgrid, world2index_offset, world2index_scale, aabb, rng, scale, acc);
}

__global__ void volume2image_convert_rays_to_spherical(uint32_t count,
    BoundingBox aabb,
    vec4* __restrict__ spherical_coordinates,
    Testbed::Volume2ImageRayInformation* __restrict__ ray_information){

    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= count)
        return;

    vec3 dir = ray_information[idx].direction;
    vec3 pos = ray_information[idx].position;

    // convert to spherical coordinates
    vec2 sperical_pos_1 = vec2(0.0f);
    vec2 sperical_pos_2 = vec2(0.0f);

    convert_ray_to_spherical_coordinates(pos, dir, aabb, sperical_pos_1, sperical_pos_2);

    // Write the spherical coordinates to the buffer
    spherical_coordinates[idx] = vec4(sperical_pos_1.x, sperical_pos_1.y, sperical_pos_2.x, sperical_pos_2.y);
}

__global__ void volume2image_move_pixels_to_framebuffer(
        uint32_t n_pixels,
        ivec2 resolution,
        vec4* __restrict__ pixels,
        const Testbed::VolPayload* __restrict__ payload,
        vec4* __restrict__ frame_buffer
) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n_pixels)
		return;

    uint32_t pixidx = payload[idx].pixidx;
	uint32_t y = pixidx / resolution.x;
	if (y >= resolution.y)
		return;

    frame_buffer[pixidx] = pixels[idx];
}

void Testbed::render_volume2image(
	cudaStream_t stream,
	const CudaRenderBufferView& render_buffer,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const Foveation& foveation
) {
	float plane_z = m_slice_plane_z + m_scale;
	float distance_scale = 1.f/std::max(m_volume.inv_distance_scale,0.01f);
	auto res = render_buffer.resolution;

	size_t n_pixels = (size_t)res.x * res.y;

    // Unlike the volume rendering we only need 1 buffer and not both
    m_volume.pos[0].enlarge(n_pixels);
    m_volume.payload[0].enlarge(n_pixels);

    // But additionally we need the buffer with the combined ray and ray direction
    m_volume2image.rays.enlarge(n_pixels);

	m_volume.hit_counter.enlarge(2);
	m_volume.hit_counter.memset(0);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1 };
	init_rays_volume<<<blocks, threads, 0, stream>>>(
		render_buffer.spp,
		m_volume.pos[0].data(),
		m_volume.payload[0].data(),
        m_volume2image.rays.data(),
		m_volume.hit_counter.data(),
		res,
		focal_length,
		camera_matrix,
		screen_center,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		m_render_aabb,
		m_render_near_distance,
		plane_z,
		m_aperture_size,
		foveation,
		render_buffer.frame_buffer,
		render_buffer.depth_buffer,
		render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
		m_rng,
		m_volume.bitgrid.data(),
		distance_scale,
		m_volume.global_majorant,
        m_volume2image.evaluate_rays_at_bounding_box
	);
	m_rng.advance(n_pixels*256);

    // Print to see if we reach this point
    //tlog::info() << "Initialized rays";

	uint32_t n=n_pixels;
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
    cudaMemcpy(&n, m_volume.hit_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost);

	if (m_render_ground_truth) {

        linear_kernel(volume2image_render_kernel_gt, 0, stream,
            n,
            res,
            m_rng,
            m_render_aabb,
            m_volume.transfer_function.data(),
            m_volume.transfer_function.size(),
            m_volume.pos[0].data(),
            m_volume.payload[0].data(),
            m_volume.hit_counter.data(),
            m_volume.nanovdb_grid.data(),
            m_volume.bitgrid.data(),
            m_volume.global_majorant,
            m_volume.world2index_offset,
            m_volume.world2index_scale,
            distance_scale,
            render_buffer.frame_buffer
        );

		m_rng.advance(n_pixels*256);
	} else {
        // Calculate how many samples we evaluate
        uint32_t n_elements = next_multiple(n, tcnn::batch_size_granularity);

        // Ensure that our pixel buffer is large enough
        m_volume2image.pixel_information.enlarge(n_elements);

        // Depending on the network mode we have different inputs and inference functions
        if (m_volume2image.mode == Testbed::EVolume2ImageMode::DefaultRay) {
            // Construct the matrices for the network inference
            GPUMatrix<float> rays_matrix((float*)m_volume2image.rays.data(), 6, n_elements);
            GPUMatrix<float> pixel_information_matrix((float*)m_volume2image.pixel_information.data(), 4, n_elements);
            // Run inference on the network
            m_network->inference(stream, rays_matrix, pixel_information_matrix);
        } else {
            // Otherwise we are in spherical mode, and first need to fetch the spherical parameters from the rays
            m_volume2image.spherical_positions.enlarge(n_elements);
            linear_kernel(volume2image_convert_rays_to_spherical, 0, stream, n, m_render_aabb, m_volume2image.spherical_positions.data(), m_volume2image.rays.data());

            // Construct the matrices for the network inference
            GPUMatrix<float> spherical_positions_matrix((float*)m_volume2image.spherical_positions.data(), 4, n_elements);
            GPUMatrix<float> pixel_information_matrix((float*)m_volume2image.pixel_information.data(), 4, n_elements);

            // Run inference on the network
            m_network->inference(stream, spherical_positions_matrix, pixel_information_matrix);
        }

        // Move the output into the framebuffer
        linear_kernel(volume2image_move_pixels_to_framebuffer, 0, stream, n, res, m_volume2image.pixel_information.data(), m_volume.payload[0].data(), render_buffer.frame_buffer);
	}
}

void Testbed::load_volume2image(const fs::path& data_path) {
    // We need exactly the same data as the original volume testbed, so we actually just call it's loading function
    // and continue using the volume construct in a limited way in this testbed
    load_volume(data_path);

    // The assumption we do make, is that a transfer function is present
    // So if is none is present, create a dummy transfer function which just maps the density to a grayish color
    if (m_volume.transfer_function.size() == 0) {
        // Have a vector of vec4 to store the transfer function
        std::vector<vec4> transfer_function;
        transfer_function.resize(256);
        for (size_t i = 0; i < 256; i++) {
            transfer_function[i] = {0.5f, 0.5f, 0.5f, i/255.0f};
        }

        // Ensure it is on the GPU
        m_volume.transfer_function.enlarge(transfer_function.size());
        m_volume.transfer_function.copy_from_host(transfer_function);
    }
}

NGP_NAMESPACE_END
