#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 1);
    uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER
#include programs/pcg_random.glsl
#include programs/utils.glsl
#line 19

in vec2 uv;

uniform vec3 uCameraPos;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform sampler2D u_depth;
uniform usampler3D u_voxel_data;
uniform sampler2D u_motion_vector;
uniform sampler2D u_last_frame;
uniform sampler2DArray u_normals;
uniform sampler2DArray u_random_vec2;
uniform sampler2D u_last_frame_depth;
uniform vec3 lightPos;
uniform int frame_counter;

const float PI = 3.14159265;
const int MAX_OCC_SAMPLES = 2;
const int MAX_OCC_DISTANCE = 40;
float LIGHT_RADIUS = 10.0;

uint rnd_seed = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * 4097U + uint(frame_counter);
int normal_rand_state = int(rnd_seed) % 64;
int light_rand_state = int(rnd_seed) % 64;
vec2 texel_size = 1.0 / vec2(textureSize(u_albedo, 0));

vec3 size = textureSize(u_voxel_data, 0);
Box bbox = Box(vec3(0.0), vec3(size));
int MAX_STEPS = int(max(size.x, max(size.y, size.z))) * 3;

out vec4 fragColor;

vec3 decodeNormalRGB10A2(vec3 encoded)
{
    // map [0,1] -> [-1,1]
    vec3 decoded = encoded * 2.0 - 1.0;
    return normalize(decoded);
}

vec3 generate_random_normal(inout int seed) {
    seed = (seed + 1) % 64;
    vec3 rnd_normal_coord = vec3(mod(gl_FragCoord.xy, 128) / 128.0, seed);
    return texture(u_normals, rnd_normal_coord).rgb * 2.0 - 1.0;
}

vec2 generate_random_vec2(inout int seed) {
    seed = (seed + 1) % 64;
    vec3 rnd_normal_coord = vec3(mod(gl_FragCoord.xy, 128) / 128.0, seed);
    return texture(u_random_vec2, rnd_normal_coord).rg;
}

vec3 tangentToWorld(vec3 surfaceNormal, vec3 normalTS) {
    vec3 N = normalize(surfaceNormal);

    // Pick an arbitrary vector not parallel to N
    vec3 up = normalize(cross(N, vec3(0.0, 1.0, 1.0)));

    // Build tangent basis
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);

    // Transform tangent-space normal into world space
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * normalTS);
}

vec3 generate_random_cosine_weighted_normal(vec3 normal, inout int seed) {
    vec3 normal_tangent = generate_random_normal(seed);
    return tangentToWorld(normal, normal_tangent);
}

vec3 reconstructWorldPos(float depth)
{
    vec2 ndc = uv * 2.0 - 1.0;
    // Depth buffer [0,1] -> NDC z [-1,1]
    float ndcZ = depth * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc, ndcZ, 1.0);
    // Clip space -> world space
    vec4 worldPos = uInvView * uInvProjection * clipPos;
    // Perspective divide
    return worldPos.xyz / worldPos.w;
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

vec3 compute_light(vec3 camera_pos, vec3 pos, vec3 normal, vec3 light_pos, vec3 albedo, Pcg32State rnd) {
    vec3 ray_start = pos + normal * 0.5;

    // Ambient Occlusion
    float ambient_gathered = 0;
    for (int occ_sample; occ_sample < MAX_OCC_SAMPLES; occ_sample += 1) {
        vec3 jitter_point = (pcg_random_vec3(rnd) - 0.5);
        vec3 jitter = jitter_point - normal * dot(jitter_point, normal);
        Ray occ_ray = Ray(ray_start + jitter, generate_random_cosine_weighted_normal(normal, normal_rand_state));
        Hit occ_hit = dda(occ_ray, MAX_OCC_DISTANCE, u_voxel_data, bbox);
        if (occ_hit.hit) {
            ambient_gathered += clamp(occ_hit.t, 0, MAX_OCC_DISTANCE);
        } else {
            ambient_gathered += MAX_OCC_DISTANCE;
        }
    }

    // Ambient from sky
    vec3 ambientColor = skyColor(normal);
    float occlusion = ambient_gathered / (MAX_OCC_SAMPLES * MAX_OCC_DISTANCE);
    occlusion *= occlusion;
    vec3 ambient = albedo * ambientColor * occlusion * 0.5;

    // Sun ray
    vec3 light_center = sample_disk_light(light_pos, normalize(pos - light_pos), LIGHT_RADIUS, generate_random_vec2(light_rand_state));
    vec3 L = normalize(light_pos - pos); // direction to light
    Ray sun_ray = Ray(ray_start, normalize(light_center - ray_start));
    Hit sun_hit = dda(sun_ray, MAX_STEPS, u_voxel_data, bbox);
    //return vec3(occlusion) + albedo * 0.00001 + camera_pos * 0.000001;
    vec3 color = ambient;
    if (!sun_hit.hit) {
        vec3 lightColor = vec3(20.0, 18.0, 15.0) * 500.0;
        float shininess = 1000.0;

        float distance = length(light_pos - pos);
        // Lambert cosine term
        float NdotL = max(dot(normal, L), 0.0);

        // Inverse square falloff
        float attenuation = 1.0 / (distance * distance);
        vec3 diffuse = albedo * lightColor * NdotL * attenuation;

        // specular
        vec3 V = normalize(camera_pos - pos);
        vec3 H = normalize(L + V);
        float NdotH = max(dot(normal, H), 0.0);
        float spec = pow(NdotH, shininess);
        vec3 specular = lightColor * spec * attenuation;

        color += diffuse + specular * 0.000001;
    }
    return color;
}

void main() {
    Pcg32State rnd = pcg_srandom(rnd_seed);

    Ray camera_ray = compute_camera_ray(uInvProjection, uInvView, uCameraPos, frame_counter, 0.0);
    float depth = texture(u_depth, uv).r;
    if (depth == 1.0) {
        fragColor = vec4(skyColor(camera_ray.direction), 1.0);
        return;
    }
    vec3 albedo = texture(u_albedo, uv).rgb;
    vec3 normal = decodeNormalRGB10A2(texture(u_normal, uv).rgb);
    vec3 pos = reconstructWorldPos(depth);
    vec3 color = compute_light(uCameraPos, pos, normal, lightPos, albedo, rnd);

    vec2 last_frame_uv = uv + texture(u_motion_vector, uv).rg;
    float last_frame_depth = texture(u_last_frame_depth, last_frame_uv).r;
    float current_depth = distance(camera_ray.origin, pos);
    bool outside = any(lessThanEqual(last_frame_uv, texel_size * 2.0)) || any(greaterThanEqual(last_frame_uv, vec2(1.0) - texel_size * 2.0));
    if (outside || abs(last_frame_depth - current_depth) > 1.0) {
        fragColor = vec4(color, 1.0);
    } else {
        vec3 last_frame_color = texture(u_last_frame, last_frame_uv).rgb;
        fragColor = vec4(mix(color, last_frame_color, 0.9), 1.0);
    }
}
#endif
