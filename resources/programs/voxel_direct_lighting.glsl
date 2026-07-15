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

#define IS_SUN 0

in vec2 uv;

uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform int frame_counter;
#if IS_SUN == 1
uniform vec3 sunDirection;
#else
uniform vec3 lightPos;
#endif
uniform float lightRadius;
uniform vec3 lightColor;

layout(binding = 0) uniform sampler2D u_normal;
layout(binding = 1) uniform sampler2D u_depth;
layout(binding = 2) uniform sampler2D u_linear_depth;
layout(binding = 3) uniform usampler3D u_voxel_data;
layout(binding = 4) uniform sampler2DArray u_stbn_vec2;

layout(location = 0) out vec3 out_irradiance;

const float PI = 3.14159265;

uint rnd_seed = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * 4097U + uint(frame_counter);
int light_rand_state = int(rnd_seed) % 64;

mat4 projectionview = uProjection * uView;
float linear_depth = texture(u_linear_depth, uv).r;
vec3 camera_pos = uInvView[3].xyz;
vec3 size = textureSize(u_voxel_data, 0);
Box bbox = Box(vec3(0.0), vec3(size));
int MAX_STEPS = int(max(size.x, max(size.y, size.z))) * 3;

vec2 generate_random_vec2(inout int seed) {
    seed = (seed + 1) % 64;
    vec3 rnd_normal_coord = vec3(mod(gl_FragCoord.xy, 128) / 128.0, seed);
    return texture(u_stbn_vec2, rnd_normal_coord).rg;
}

vec3 sample_disk_light(vec3 lightPos, vec3 lightNormal, float radius, vec2 xi) {
    float r = radius * sqrt(xi.x);
    float phi = 2.0 * PI * xi.y;

    vec3 T = normalize(cross(lightNormal, abs(lightNormal.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0)));
    vec3 B = cross(lightNormal, T);

    return lightPos +
        T * (r * cos(phi)) +
        B * (r * sin(phi));
}

vec3 compute_direct_light(vec3 pos, vec3 normal, vec3 light_pos) {
    vec3 ray_start = pos + normal * 0.1;

    // Shadow ray
    vec3 light_center = sample_disk_light(light_pos, normalize(pos - light_pos), lightRadius, generate_random_vec2(light_rand_state));
    vec3 L = normalize(light_pos - pos); // direction to light
    Ray sun_ray = Ray(ray_start, normalize(light_center - ray_start));
    Hit sun_hit = dda(sun_ray, MAX_STEPS, u_voxel_data, bbox);
    if (!sun_hit.hit || sun_hit.t >= distance(pos, light_pos)) {
        float distance = length(light_pos - pos);
        // Lambert cosine term
        float NdotL = max(dot(normal, L), 0.0);

        // Inverse square falloff
        float attenuation = 1.0 / (distance * distance);
        vec3 diffuse = lightColor * NdotL * attenuation;
        return diffuse;
    }
    return vec3(0.0);
}

vec3 compute_direct_sun(vec3 pos, vec3 normal, vec3 sun_direction) {
    vec3 ray_start = pos + normal * 0.1;

    // Shadow ray
    vec3 L = normalize(sample_disk_light(sun_direction, normalize(sun_direction), lightRadius, generate_random_vec2(light_rand_state))); // direction to light

    Ray sun_ray = Ray(ray_start, L);
    Hit sun_hit = dda(sun_ray, MAX_STEPS, u_voxel_data, bbox);
    if (!sun_hit.hit) {
        // Lambert cosine term
        float NdotL = max(dot(normal, L), 0.0);
        vec3 diffuse = lightColor * NdotL;
        return diffuse;
    }
    return vec3(0.0);
}

void main() {
    Ray camera_ray = compute_camera_ray(uv, uInvProjection, uInvView, 0, 0.0);
    float depth = texture(u_depth, uv).r;
    if (depth == 1.0) {
        return;
    }
    vec3 normal = decodeNormalRGB10A2(texture(u_normal, uv).rgb);
    vec3 pos = camera_ray.origin + camera_ray.direction * linear_depth;
    #if IS_SUN == 1
    vec3 color = compute_direct_sun(pos, normal, sunDirection);
    #else
    vec3 color = compute_direct_light(pos, normal, lightPos);
    #endif

    out_irradiance = color;
}
#endif
