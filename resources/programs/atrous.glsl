#version 420

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 1);
    uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER
#include programs/utils.glsl
#line 18

#define CAMERA_FAR

in vec2 uv;

layout(binding = 0) uniform sampler2D tex_current;
layout(binding = 1) uniform sampler2D tex_current_depth;
layout(binding = 2) uniform sampler2D tex_current_normals;
uniform float step_size;
uniform float depth_sigma = 0.5;
uniform float normal_sigma = 0.1;

layout(location = 0) out vec3 clean_color;

vec2 texel_size = 1.0 / vec2(textureSize(tex_current, 0));
const float gaussian_kernel[25] = float[](
        1.0, 4.0, 6.0, 4.0, 1.0,
        4.0, 16.0, 24.0, 16.0, 4.0,
        6.0, 24.0, 36.0, 24.0, 6.0,
        4.0, 16.0, 24.0, 16.0, 4.0,
        1.0, 4.0, 6.0, 4.0, 1.0
    );

void main() {
    float weight_sum = 0.0;
    vec3 color = vec3(0.0);
    int kernel_index = 0;
    float current_depth = texture(tex_current_depth, uv).r;
    vec3 current_normal = texture(tex_current_normals, uv).rgb;
    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec2 sample_uv = uv + vec2(x, y) * texel_size * step_size;
            if (any(lessThan(sample_uv, vec2(0))) || any(greaterThan(sample_uv, vec2(1)))) {
                continue;
            }

            float depth_diff = abs(current_depth - texture(tex_current_depth, sample_uv).r);
            float depth_weight = exp(-depth_diff * depth_diff / (2.0 * depth_sigma * depth_sigma));
            float normal_diff = 1.0 - max(0.0, dot(current_normal, texture(tex_current_normals, sample_uv).rgb));
            float normal_weight = exp(-normal_diff * normal_diff / (2.0 * normal_sigma * normal_sigma));
            float weight = gaussian_kernel[kernel_index] * depth_weight * normal_weight;
            kernel_index++;

            color += texture(tex_current, sample_uv).rgb * weight;
            weight_sum += weight;
        }
    }
    clean_color = color / weight_sum;
}
#endif
