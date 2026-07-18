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

in vec2 uv;

uniform mat4 u_inv_projection;
uniform mat4 u_inv_view;
uniform int frame_counter;
uniform bool use_history_clamping;

layout(binding = 0) uniform sampler2D tex_last;
layout(binding = 1) uniform sampler2D tex_current;
layout(binding = 2) uniform sampler2D tex_motion_vectors;
layout(binding = 3) uniform sampler2D tex_current_depth;
layout(binding = 4) uniform sampler2D tex_current_normals;
layout(binding = 5) uniform sampler2DArray stbn_scalar;
layout(binding = 6) uniform sampler2D tex_last_depth;

layout(location = 0) out vec3 clean_color;

const float PI = 3.14159265;
const float GOLDEN_RATIO = 2.39996;
vec2 texel_size = 1.0 / vec2(textureSize(tex_current, 0));
uint rnd_seed = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * 4097U + uint(frame_counter);
int rotation_random_state = int(rnd_seed) % 64;

struct Plane {
    float d;
    vec3 normal;
};

float distance_to_plane(Ray ray, Plane plane) {
    // Ray: X = O + t * D
    // Plane: N * X = d
    // -> N * (O + t * D) = d
    // -> N * O + t * N * D = d
    // -> t = (d - N * O) / (N * D)
    // TODO(david): check if denominator is smaller than epsilon for grazing angles
    return (plane.d - dot(plane.normal, ray.origin)) / dot(plane.normal, ray.direction);
}

float blue_noise_random_num() {
    return 0.0;
}

vec3 spiral_sampling(vec2 uv, Plane plane, vec3 current_normal, vec3 current_color, inout vec3 minimum_color, inout vec3 maximum_color) {
    vec3 accumulated_color = current_color;
    float weight = 1;
    float rotation = generate_random_stbn_scalar(stbn_scalar, rotation_random_state) * PI * 2;
    for (int i = 1; i < 12; i++) {
        float radius = float(i);
        rotation += GOLDEN_RATIO;
        vec2 offset = radius * texel_size * vec2(cos(rotation), sin(rotation));
        vec2 sample_uv = uv + offset;
        vec3 neighbor_col = texture(tex_current, sample_uv).rgb;
        float neighbor_depth = texture(tex_current_depth, sample_uv).r;
        vec3 neighbor_normal = texture(tex_current_normals, sample_uv).rgb;
        Ray neighbor_ray = compute_camera_ray(sample_uv, u_inv_projection, u_inv_view, 0, 0.0);
        if (abs(distance_to_plane(neighbor_ray, plane) - neighbor_depth) > 0.1) {
            continue;
        }
        if (dot(current_normal, neighbor_normal) < 0.9) {
            continue;
        }
        float sample_weight = 1.2 - radius / 12.0;
        vec3 filtered_sample = mix(current_color, neighbor_col, sample_weight);
        minimum_color = min(minimum_color, filtered_sample);
        maximum_color = max(maximum_color, filtered_sample);
        accumulated_color += neighbor_col * sample_weight;
        weight += sample_weight;
    }
    vec3 resolved_color = accumulated_color / weight;
    return resolved_color;
}

void main() {
    float current_depth = texture(tex_current_depth, uv).r;
    vec3 current_color = texture(tex_current, uv).rgb;
    if (current_depth > 1000) {
        clean_color = current_color;
        return;
    }
    float depth_below = texture(tex_current_depth, uv + vec2(0, -texel_size.y)).r;
    vec2 uv_motion = uv;
    vec2 motion_vector = texture(tex_motion_vectors, uv_motion).rg;
    vec2 old_uv = uv + motion_vector;
    vec3 current_normal = texture(tex_current_normals, uv).rgb;
    Ray ray = compute_camera_ray(uv, u_inv_projection, u_inv_view, 0, 0.0);
    vec3 world_space_pos = ray.origin + ray.direction * current_depth;
    float plane_d = dot(world_space_pos, current_normal);
    Plane plane = Plane(plane_d, current_normal);
    vec3 minimum_color = current_color;
    vec3 maximum_color = current_color;
    vec3 resolved_color = spiral_sampling(uv, plane, current_normal, current_color, minimum_color, maximum_color);
    vec3 history_color = texture(tex_last, old_uv).rgb;
    float last_depth = texture(tex_last_depth, old_uv).r;
    // reject history
    if (any(lessThan(old_uv, vec2(0))) || any(greaterThan(old_uv, vec2(1))) || abs(last_depth - current_depth) > 3.0) {
        clean_color = resolved_color;
        return;
    }
    if (use_history_clamping) {
        history_color = clamp(history_color, minimum_color, maximum_color);
    }
    float blend_factor = 0.7 - clamp(length(motion_vector) * 0.4, 0.0, 0.4);
    clean_color = mix(resolved_color, history_color, blend_factor);
}
#endif
