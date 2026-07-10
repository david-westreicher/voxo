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
uniform float time;
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform sampler2D u_depth;
uniform usampler3D u_voxel_data;

const float PI = 3.14159265;
const int MAX_OCC_SAMPLES = 10;
const int MAX_OCC_DISTANCE = 20;

Box bbox = compute_bbox(u_voxel_data);
vec3 size = bbox.max - bbox.min;
int MAX_STEPS = int(max(size.x, max(size.y, size.z))) * 3;

vec3 lightPos = vec3(sin(time * 0.01), 0.5, cos(time * 0.01)) * MAX_STEPS * 2.0;

out vec4 fragColor;

vec3 decodeNormalRGB10A2(vec3 encoded)
{
    // map [0,1] -> [-1,1]
    vec3 decoded = encoded * 2.0 - 1.0;
    return normalize(decoded);
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

vec3 cosineSampleHemisphere(vec3 n, vec2 u) {
    //TODO(david): Use precomputed blue noise samples
    float r = sqrt(u.x);
    float theta = 2.0 * PI * u.y;

    vec3 B = normalize(cross(n, vec3(0.0, 1.0, 1.0)));
    vec3 T = cross(B, n);

    return normalize(r * sin(theta) * B + sqrt(1.0 - u.x) * n + r * cos(theta) * T);
}

vec3 compute_light(vec3 camera_pos, vec3 pos, vec3 normal, vec3 albedo, Pcg32State rnd) {
    vec3 ray_start = pos + normal * 0.1;

    // Ambient Occlusion
    float ambient_gathered = 0;
    for (int occ_sample; occ_sample < MAX_OCC_SAMPLES; occ_sample += 1) {
        vec3 jitter_point = (pcg_random_vec3(rnd) - 0.5);
        vec3 jitter = jitter_point - normal * dot(jitter_point, normal);
        Ray occ_ray = Ray(ray_start + jitter, cosineSampleHemisphere(normal, pcg_random_vec2(rnd)));
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
    vec3 ambient = albedo * ambientColor * occlusion;

    // Sun ray
    vec3 L = normalize(lightPos - pos); // direction to light
    Ray sun_ray = Ray(ray_start, L);
    Hit sun_hit = dda(sun_ray, MAX_STEPS, u_voxel_data, bbox);
    //return vec3(occlusion) + albedo * 0.00001 + camera_pos * 0.000001;
    if (sun_hit.hit) {
        return ambient;
    } else {
        vec3 lightColor = vec3(20.0, 18.0, 15.0) * MAX_STEPS * MAX_STEPS * 0.5;
        float shininess = 60;

        float distance = length(lightPos - pos);
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

        return ambient + diffuse + specular;
    }
}

void main() {
    Pcg32State rnd = pcg_srandom(uint(gl_FragCoord.x) +
                uint(gl_FragCoord.y) * 4097U +
                uint(time) * 1234567U);

    Ray camera_ray = compute_camera_ray(uInvProjection, uInvView, uCameraPos);
    float depth = texture(u_depth, uv).r;
    if (depth == 1.0) {
        fragColor = vec4(skyColor(camera_ray.direction), 1.0);
        return;
    }
    vec3 albedo = texture(u_albedo, uv).rgb;
    vec3 normal = decodeNormalRGB10A2(texture(u_normal, uv).rgb);
    vec3 pos = reconstructWorldPos(depth);
    vec3 color = compute_light(uCameraPos, pos, normal, albedo, rnd);
    fragColor = vec4(color, 1.0);
}
#endif
