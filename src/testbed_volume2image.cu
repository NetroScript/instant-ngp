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

#include <filesystem>
#include <stb_image/stb_image.h>

#include <fstream>

using namespace tcnn;

NGP_NAMESPACE_BEGIN

Testbed::NetworkDims Testbed::network_dims_volume2image() const {
	NetworkDims dims;
	dims.n_input = 6;
	dims.n_output = 4;
	dims.n_pos = 6;
	return dims;
}

// Sample a transfer function
// The transfer function is a 1D array of vec4 values, where each value is a color and opacity.
// The density value is a float in the range [0, 1], and the transfer function is sampled
// linearly between the two closest values.
__host__ __device__ vec4 sample_transfer_function(const vec4* transfer_function, const int transfer_function_size, float density) {
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

void Testbed::train_volume2image(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	const uint32_t n_output_dims = 4;
	const uint32_t n_input_dims = 6;

	// Auxiliary matrices for training
	const uint32_t batch_size = (uint32_t)target_batch_size;

	// Permute all training records to de-correlate training data

	const uint32_t n_elements = batch_size;
	m_volume2image.training.rays.enlarge(n_elements);
    m_volume2image.training.colors.enlarge(n_elements);

	float distance_scale = 1.f/std::max(m_volume.inv_distance_scale,0.01f);


    // Use the transfer function
    //linear_kernel(, 0, stream, n_elements / MAX_TRAIN_VERTICES, );

	m_rng.advance(n_elements*256);

	GPUMatrix<float> training_batch_matrix((float*)(m_volume2image.training.rays.data()), n_input_dims, batch_size);
	GPUMatrix<float> training_target_matrix((float*)(m_volume2image.training.colors.data()), n_output_dims, batch_size);

	auto ctx = m_trainer->training_step(stream, training_batch_matrix, training_target_matrix);

	m_training_step++;

	if (get_loss_scalar) {
		m_loss_scalar.update(m_trainer->loss(stream, *ctx));
	}
}

__global__ void init_rays_volume(
	uint32_t sample_index,
	vec3* __restrict__ positions,
	Testbed::VolPayload* __restrict__ payloads,
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
	float global_majorant
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

	if (t >= box_intersection.y || !walk_to_next_event(rng, aabb, ray.o, ray.d, bitgrid, scale)) {
		frame_buffer[idx] = vec4(0.0f ,0.0f, 0.0f, 1.0f);
		depth_buffer[idx] = MAX_DEPTH();
	} else {
		uint32_t dstidx = atomicAdd(pixel_counter, 1);
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

    // We go front to back, so we start with 100% opacity
    vec4 col = {0.0f, 0.0f, 0.0f, 1.0f};

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

        if (!walk_to_next_event(rng, aabb, pos, dir, bitgrid, scale))
            break;
    }

    // As we are doing front to back we now need to invert the alpha channel
    col.a = 1.f - col.a;

    // the ray is done!

    frame_buffer[pixidx] = col;
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
	for (uint32_t i=0;i<2;++i) {
		m_volume.pos[i].enlarge(n_pixels);
		m_volume.payload[i].enlarge(n_pixels);
	}
	m_volume.hit_counter.enlarge(2);
	m_volume.hit_counter.memset(0);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1 };
	init_rays_volume<<<blocks, threads, 0, stream>>>(
		render_buffer.spp,
		m_volume.pos[0].data(),
		m_volume.payload[0].data(),
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
		m_volume.global_majorant
	);
	m_rng.advance(n_pixels*256);

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
		// TODO: Add network evaluation and rendering
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
