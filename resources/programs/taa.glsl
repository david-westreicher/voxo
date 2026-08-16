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
#line 18 3

in vec2 uv;

layout(binding = 0) uniform sampler2D u_image;
layout(binding = 1) uniform sampler2D u_last_image;
layout(binding = 2) uniform sampler2D u_motion_vectors;
layout(binding = 3) uniform sampler2D u_linear_depth;
layout(binding = 4) uniform sampler2D u_prev_linear_depth;
layout(binding = 5) uniform sampler2D u_reflectivity;

layout(location = 0) out vec3 fragColor;

const vec2 plus_offsets[] = vec2[](
        vec2(0.0, 0.0),
        vec2(-1.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, -1.0),
        vec2(0.0, 1.0)
    );
vec2 texel_size = 1 / textureSize(u_image, 0).xy;

void min_max_color_neighborhood(sampler2D u_image, vec2 uv, inout vec3 min_color, inout vec3 max_color) {
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec3 tex_sample = texture(u_image, uv + vec2(x, y) * texel_size).rgb;
            min_color = min(min_color, tex_sample);
            max_color = max(max_color, tex_sample);
        }
    }
}

void min_max_color_plus(sampler2D u_image, vec2 uv, inout vec3 min_color, inout vec3 max_color) {
    for (int i = 0; i < 5; ++i) {
        vec3 tex_sample = texture(u_image, uv + plus_offsets[i] * texel_size).rgb;
        min_color = min(min_color, tex_sample);
        max_color = max(max_color, tex_sample);
    }
}

vec2 closest_fragment(vec2 uv) {
    float minimum_depth = 10000.0;
    vec2 closest = uv;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 coord = uv + vec2(x, y) * texel_size;
            float depth = texture(u_linear_depth, coord).x;
            if (depth < minimum_depth) {
                depth = minimum_depth;
                closest = coord;
            }
        }
    }
    return closest;
}

vec3 clip_aabb(vec3 color, vec3 minimum, vec3 maximum) {
    // Note: only clips towards aabb center (but fast!)
    vec3 center = 0.5 * (maximum + minimum);
    vec3 extents = 0.5 * (maximum - minimum);

    // This is actually `distance`, however the keyword is reserved
    vec3 offset = color - center;

    vec3 ts = abs(extents / (offset + 0.0001));
    float t = clamp(min(min(ts.x, ts.y), ts.z), 0.0, 1.0);
    return center + offset * t;
}

vec3 clip_color(vec3 current_color, vec3 last_color) {
    vec3 curr_min_neigh_color = current_color;
    vec3 curr_max_neigh_color = current_color;
    min_max_color_neighborhood(u_image, uv, curr_min_neigh_color, curr_max_neigh_color);
    vec3 curr_min_plus_color = current_color;
    vec3 curr_max_plus_color = current_color;
    min_max_color_plus(u_image, uv, curr_min_plus_color, curr_max_plus_color);
    vec3 curr_min_color = (curr_min_neigh_color + curr_min_plus_color) * 0.5;
    vec3 curr_max_color = (curr_max_neigh_color + curr_max_plus_color) * 0.5;

    return clip_aabb(last_color, curr_min_color, curr_max_color);
}

void main() {
    vec2 motion_vector = texture(u_motion_vectors, closest_fragment(uv)).rg;
    vec3 current_color = texture(u_image, uv).rgb;
    vec2 old_uv = uv + motion_vector;
    vec3 last_color = texture(u_last_image, old_uv).rgb;
    if (length(motion_vector * SCREEN_DIMENSIONS) < 0.01) {
        fragColor = mix(current_color, last_color, 0.95);
        return;
    }
    float reflectivity = texture(u_reflectivity, uv).r;
    if (motion_vector.x == -2.0 || any(lessThan(old_uv, vec2(0))) || any(greaterThan(old_uv, vec2(1))) || reflectivity > 0.5) {
        fragColor = current_color;
        return;
    }

    vec3 last_color_clipped = clip_color(current_color, last_color);
    float current_depth = texture(u_linear_depth, uv).r;
    float last_depth = texture(u_prev_linear_depth, old_uv).r;
    float depth_difference = clamp(abs(current_depth - last_depth) - current_depth * 0.1, 0.0, 1.0);
    float history_weight = min(0.95, smoothstep(0.0, 0.01, 1.0 - depth_difference));
    vec3 mixed_history = mix(current_color, last_color, history_weight);
    vec3 mixed_history_clamped = mix(current_color, last_color_clipped, 0.95);
    fragColor = mix(mixed_history_clamped, mixed_history, history_weight);
}
#endif
