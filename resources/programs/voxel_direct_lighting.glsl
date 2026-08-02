#version 420

#define LIGHT_TYPE
#define LIGHT_TYPE_SUN    0
#define LIGHT_TYPE_SPHERE 1
#define LIGHT_TYPE_CONE   2
#define LIGHT_TYPE_AREA   3

#if defined VERTEX_SHADER

in vec3 in_position;

#if LIGHT_TYPE == LIGHT_TYPE_SUN

// sun is a fullscreen pass
void main() {
    gl_Position = vec4(in_position, 1);
}

#else

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

void main() {
    mat4 m_view = m_camera * m_model;
    vec4 p = m_view * vec4(in_position, 1.0);
    gl_Position = m_proj * p;
}

#endif

#elif defined FRAGMENT_SHADER

#include programs/utils.glsl
#line 33 3

#if LIGHT_TYPE == LIGHT_TYPE_SUN
uniform vec3 sunDirection;
#else
uniform vec3 lightPos;
uniform float reach;
uniform float unshadowed;
#endif

#if LIGHT_TYPE == LIGHT_TYPE_CONE
uniform vec3 lightDirection;
uniform float penumbraCos;
#endif

#if LIGHT_TYPE == LIGHT_TYPE_AREA
uniform mat3 light_matrix;
#endif

uniform int frame_counter;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform float lightRadius;
uniform vec3 lightColor;
uniform ivec3 occluder_translation;

layout(binding = 0) uniform sampler2D u_normal;
layout(binding = 1) uniform sampler2D u_depth;
layout(binding = 2) uniform sampler2D u_linear_depth;
layout(binding = 3) uniform usampler3D u_voxel_data;
layout(binding = 4) uniform sampler2DArray u_stbn_vec2;

layout(location = 0) out vec3 out_irradiance;

const float PI = 3.14159265;

vec2 uv = gl_FragCoord.xy / SCREEN_DIMENSIONS;
vec3 size = textureSize(u_voxel_data, 0);
Box bbox = Box(vec3(0.0), vec3(size));
uint rnd_seed = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * 4097U + uint(frame_counter);
int light_rand_state = int(rnd_seed) % 64;

vec3 sample_disk_light(vec3 lightPos, vec3 lightNormal, float radius, vec2 xi) {
    float r = radius * sqrt(xi.x);
    float phi = 2.0 * PI * xi.y;

    vec3 T = normalize(cross(lightNormal, abs(lightNormal.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0)));
    vec3 B = cross(lightNormal, T);

    return lightPos +
        T * (r * cos(phi)) +
        B * (r * sin(phi));
}

vec3 sample_area_light(mat3 light_matrix, vec2 rand_vec) {
    return light_matrix * vec3(rand_vec - 0.5, 1.0);
}

#if LIGHT_TYPE == LIGHT_TYPE_SUN
vec3 compute_direct_sun(vec3 pos, vec3 normal, vec3 sun_direction) {
    vec3 ray_start = pos + normal * 1.0;
    int max_steps = int(max(size.x, max(size.y, size.z))) * 3;

    vec3 L = normalize(sample_disk_light(sun_direction, normalize(sun_direction), lightRadius, generate_random_stbn_vec2(u_stbn_vec2, light_rand_state)));

    Ray sun_ray = Ray(ray_start, L);
    sun_ray.origin -= occluder_translation;
    Hit sun_hit = sparse_raymarch(sun_ray, max_steps, u_voxel_data, bbox, 16);
    if (!sun_hit.hit) {
        // Lambert cosine term
        float NdotL = max(dot(normal, L), 0.0);
        vec3 diffuse = lightColor * NdotL;
        return diffuse;
    }
    return vec3(0.0);
}
#else
vec3 compute_direct_light(vec3 pos, vec3 normal, vec3 light_pos) {
    vec3 ray_start = pos + normal * 1.0;
    float LdotN = 1.0;

    #if LIGHT_TYPE == LIGHT_TYPE_SPHERE
    vec3 light_sample_pos = sample_disk_light(light_pos, normalize(pos - light_pos), lightRadius, generate_random_stbn_vec2(u_stbn_vec2, light_rand_state));
    #elif LIGHT_TYPE == LIGHT_TYPE_CONE
    vec3 light_sample_pos = light_pos;
    #elif LIGHT_TYPE == LIGHT_TYPE_AREA
    vec3 light_sample_pos = sample_area_light(light_matrix, generate_random_stbn_vec2(u_stbn_vec2, light_rand_state));
    #endif

    vec3 L = normalize(light_sample_pos - pos);

    #if LIGHT_TYPE == LIGHT_TYPE_AREA
    vec3 area_light_normal = normalize(cross(light_matrix[0], light_matrix[1]));
    LdotN = dot(L, -area_light_normal);
    if (LdotN <= 0) {
        return vec3(0.0);
    }
    #endif

    #if LIGHT_TYPE == LIGHT_TYPE_CONE
    LdotN = dot(L, lightDirection);
    if (LdotN <= penumbraCos)
        return vec3(0.0);
    #endif

    Ray sun_ray = Ray(ray_start, normalize(light_sample_pos - ray_start));
    sun_ray.origin -= occluder_translation;
    float sample_distance = distance(pos, light_sample_pos);
    if (sample_distance >= reach) {
        return vec3(0.0);
    }
    Hit sun_hit = sparse_raymarch(sun_ray, sample_distance, u_voxel_data, bbox, 16);
    if (!sun_hit.hit || sun_hit.t >= sample_distance - unshadowed) {
        // Lambert cosine term
        float NdotL = max(dot(normal, L), 0.0);

        float attenuation = 1.0 - smoothstep(0.0, reach, sample_distance);
        vec3 diffuse = lightColor * NdotL * LdotN * attenuation;
        return diffuse;
    }
    return vec3(0.0);
}
#endif

void main() {
    float depth = texture(u_depth, uv).r;
    if (depth == 1.0) {
        out_irradiance = vec3(0.0);
        return;
    }
    Ray camera_ray = compute_camera_ray(uv, uInvProjection, uInvView, 0, 0.0);
    float linear_depth = texture(u_linear_depth, uv).r;
    vec3 pos = camera_ray.origin + camera_ray.direction * linear_depth;
    vec3 normal = texture(u_normal, uv).rgb;
    #if LIGHT_TYPE == LIGHT_TYPE_SUN
    vec3 color = compute_direct_sun(pos, normal, sunDirection);
    #else
    vec3 color = compute_direct_light(pos, normal, lightPos);
    #endif

    out_irradiance = color;
}
#endif
