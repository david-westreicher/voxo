#version 420

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

out vec3 v_world_pos;

void main() {
    vec4 world_pos = m_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    vec4 p = m_camera * world_pos;
    gl_Position = m_proj * p;
}

#elif defined FRAGMENT_SHADER
#include programs/utils.glsl
# line 23 3

uniform vec3 u_color;
uniform vec3 u_camera_pos;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform int u_frame_counter;

in vec3 v_world_pos;

layout(binding = 0) uniform sampler2D u_in_linear_depth;
layout(binding = 1) uniform sampler2D u_in_albedo;
layout(binding = 2) uniform sampler2D u_in_water_normal_1;
layout(binding = 3) uniform sampler2D u_in_water_normal_2;

layout(location = 0) out vec3 u_albedo;
layout(location = 1) out vec3 u_normal;
layout(location = 2) out float u_linear_depth;
layout(location = 3) out vec4 u_material;
layout(location = 4) out vec2 u_motion_vector;

float worldPosToDepth(vec3 worldPos) {
    mat4 viewProj = m_proj * m_camera;
    vec4 clipPos = viewProj * vec4(worldPos, 1.0);
    float ndcDepth = clipPos.z / clipPos.w;
    // OpenGL NDC z [-1, 1] -> depth buffer [0, 1]
    return ndcDepth * 0.5 + 0.5;
}

vec3 water_normal(vec3 world_position, float time) {
    vec2 world_xz = world_position.xz * 0.005;
    vec2 uv1 = world_xz + vec2(0.03, 0.01) * time;
    vec2 uv2 = world_xz + vec2(-0.02, 0.04) * time;

    vec3 n1 = texture(u_in_water_normal_1, uv1).xyz * 2.0 - 1.0;
    vec3 n2 = texture(u_in_water_normal_2, uv2).xyz * 2.0 - 1.0;
    vec3 n = normalize(n1 + n2);

    return normalize(vec3(n.x, n.z, n.y));
}

void main() {
    vec2 uv = gl_FragCoord.xy / SCREEN_DIMENSIONS;
    float prev_linear_depth = texture(u_in_linear_depth, uv).r;
    float current_linear_depth = distance(v_world_pos, u_camera_pos);
    float water_depth = clamp(abs(prev_linear_depth - current_linear_depth) / 100.0, 0.0, 1.0);
    vec3 water_color = mix(u_color, u_color * 0.5, water_depth);
    vec3 water_norm = water_normal(v_world_pos, u_frame_counter * 0.005);
    vec3 final_normal = normalize(mix(vec3(0, 1, 0), water_norm, 0.3));
    vec2 refracted_uv = (final_normal.xz / SCREEN_DIMENSIONS) * 100.0;
    vec3 prev_albedo = texture(u_in_albedo, uv + refracted_uv).rgb;

    u_albedo = mix(prev_albedo * water_color, water_color, water_depth * 0.9);
    u_normal = final_normal;
    u_linear_depth = current_linear_depth;
    u_motion_vector = vec2(-2.0, 0.0);
    u_material = vec4(0.1, 0.0, 0.1, 0.0);
    gl_FragDepth = worldPosToDepth(v_world_pos);
}
#endif
