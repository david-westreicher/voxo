#version 400

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

void main() {
    mat4 m_view = m_camera * m_model;
    vec4 p = m_view * vec4(in_position, 1.0);
    gl_Position = m_proj * p;
}

#elif defined FRAGMENT_SHADER

#include programs/pcg_random.glsl

uniform vec3 uCameraPos;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform usampler3D u_voxel_data;
uniform sampler2D u_palette_data;
uniform float time;

in vec2 vUV;
out vec4 fragColor;

const float PI = 3.14159265;
const int MAX_OCC_SAMPLES = 10;
const int MAX_OCC_DISTANCE = 10;
vec3 size = ceil(textureSize(u_voxel_data, 0) * 0.5) * 2.0; // NOTE(david): hack to make odd dimensions work
float inv_palette_size = 1.0 / (textureSize(u_palette_data, 0).r - 1.0);
int MAX_STEPS = int(max(size.x, max(size.y, size.z))) * 3;

vec3 boxMin = -size * 0.5;
vec3 boxMax = size * 0.5;
vec3 lightPos = vec3(sin(time * 0.01), 0.5, cos(time * 0.01)) * MAX_STEPS * 2.0;
//vec3 lightPos = vec3(1.0, 0.5, cos(time * 0.01)) * MAX_STEPS * 2.0;

uint voxelmap(vec3 p)
{
    vec3 local_coord = (p - boxMin + 0.5) / vec3(size);
    return texture(u_voxel_data, local_coord).r;
}

bool is_inside_box(vec3 p) {
    return all(greaterThanEqual(p, boxMin)) &&
        all(lessThan(p, boxMax));
}

vec3 skyColor(vec3 rayDir) {
    vec3 dir = normalize(rayDir);

    // Map y from [-1,1] to [0,1]
    float t = 0.5 * (dir.y + 1.0);

    vec3 horizon = vec3(0.8, 0.9, 1.0);
    vec3 zenith = vec3(0.2, 0.4, 0.8);

    return mix(horizon, zenith, t) * 2.5;
}

vec3 phongLight(
    vec3 normal,
    vec3 lightDir,
    vec3 viewDir,
    vec3 lightColor,
    vec3 albedo,
    float shininess
) {
    vec3 N = normalize(normal);
    vec3 L = normalize(lightDir);
    vec3 V = normalize(viewDir);

    // Diffuse (Lambert)
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * lightColor * NdotL;

    // Specular (Phong)
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(R, V), 0.0), shininess);
    vec3 specular = lightColor * spec;

    return diffuse + specular;
}

struct Hit {
    bool hit;
    float t;
    vec3 position;
    vec3 voxel;
    vec3 normal;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

bool intersectAABB(
    Ray ray,
    vec3 boxMin,
    vec3 boxMax,
    out float tHit
) {
    vec3 invDir = 1.0 / ray.direction;

    vec3 t0 = (boxMin - ray.origin) * invDir;
    vec3 t1 = (boxMax - ray.origin) * invDir;

    vec3 tMin = min(t0, t1);
    vec3 tMax = max(t0, t1);

    float tNear = max(max(tMin.x, tMin.y), tMin.z);
    float tFar = min(min(tMax.x, tMax.y), tMax.z);

    // No intersection, or box is behind ray
    if (tNear > tFar || tFar < 0.0)
        return false;

    tHit = max(tNear, 0.0);
    return true;
}

Hit dda(Ray ray, int max_steps) {
    vec3 pos = ray.origin;
    vec3 rayDir = ray.direction;
    Hit hit;
    hit.hit = false;

    vec3 map = floor(pos);
    vec3 stepDir = vec3(0);
    vec3 sideDist = vec3(9e9);
    vec3 deltaDist = 1. / abs(rayDir);
    float side = 0.;
    vec3 S = step(0., rayDir);

    stepDir = 2. * S - 1.;
    sideDist = (S - stepDir * fract(pos)) * deltaDist;

    bool has_entered = is_inside_box(map);
    for (int i = 0; i < max_steps; i++) {
        vec4 conds = step(sideDist.xxyy, sideDist.yzzx);
        vec3 cases = vec3(0);
        cases.x = conds.x * conds.y;
        cases.y = (1. - cases.x) * conds.z * conds.w;
        cases.z = (1. - cases.x) * (1. - cases.y);
        sideDist += max((2. * cases - 1.) * deltaDist, 0.);
        map += cases * stepDir;
        if (is_inside_box(map) && !has_entered) {
            has_entered = true;
        }
        if (has_entered && !is_inside_box(map)) {
            return hit;
        }
        if (has_entered && voxelmap(map) > 0.) // Did we hit anything? if so, we are done!
        {
            side = cases.y + 2. * cases.z;
            break;
        }
    }
    if (!has_entered) {
        return hit;
    }
    vec3 normal = vec3(0.0);
    normal[int(side)] = -1. * sign(rayDir[int(side)]); // voxel face debug
    vec3 p = map + .5 - stepDir * .5; // Point on axis plane
    float t = (dot(normal, p - pos)) / dot(normal, rayDir);

    hit.hit = true;
    hit.t = t;
    hit.position = pos + rayDir * t;
    hit.voxel = map;
    hit.normal = normal;
    return hit;
}

vec3 cosineSampleHemisphere(vec3 n, vec2 u)
{
    float r = sqrt(u.x);
    float theta = 2.0 * PI * u.y;

    vec3 B = normalize(cross(n, vec3(0.0, 1.0, 1.0)));
    vec3 T = cross(B, n);

    return normalize(r * sin(theta) * B + sqrt(1.0 - u.x) * n + r * cos(theta) * T);
}

vec3 compute_light(Ray camera_ray, vec3 pos, vec3 normal, vec3 albedo, Pcg32State rnd) {
    vec3 ray_start = pos + normal * 0.1;

    // Ambient Occlusion
    float ambient_gathered = 0;
    for (int occ_sample; occ_sample < MAX_OCC_SAMPLES; occ_sample += 1) {
        vec3 jitter_point = (pcg_random_vec3(rnd) - 0.5);
        vec3 jitter = jitter_point - normal * dot(jitter_point, normal);
        Ray occ_ray = Ray(ray_start + jitter, cosineSampleHemisphere(normal, pcg_random_vec2(rnd)));
        Hit occ_hit = dda(occ_ray, MAX_OCC_DISTANCE);
        if (occ_hit.hit) {
            ambient_gathered += clamp(occ_hit.t, 0, MAX_OCC_DISTANCE);
        } else {
            ambient_gathered += MAX_OCC_DISTANCE;
        }
    }

    // Ambient from sky
    vec3 ambientColor = skyColor(normal);
    float occlusion = ambient_gathered / (MAX_OCC_SAMPLES * MAX_OCC_DISTANCE);
    vec3 ambient = albedo * ambientColor * occlusion;

    // Sun ray
    vec3 L = normalize(lightPos - pos); // direction to light
    Ray sun_ray = Ray(ray_start, L);
    Hit sun_hit = dda(sun_ray, MAX_STEPS);
    if (sun_hit.hit) {
        return ambient;
    } else {
        vec3 lightColor = vec3(20.0, 18.0, 15.0) * pow(MAX_STEPS, 2.0) * 1.0;
        float shininess = 60;

        float distance = length(lightPos - pos);
        // Lambert cosine term
        float NdotL = max(dot(normal, L), 0.0);

        // Inverse square falloff
        float attenuation = 1.0 / (distance * distance);
        vec3 diffuse = albedo * lightColor * NdotL * attenuation;

        // specular
        vec3 V = normalize(camera_ray.origin - pos);
        vec3 H = normalize(L + V);
        float NdotH = max(dot(normal, H), 0.0);
        float spec = pow(NdotH, shininess);
        vec3 specular = lightColor * spec * attenuation;

        return ambient + diffuse + specular;
    }
}

void main() {
    vec2 ndc = gl_FragCoord.xy / vec2(1920, 1080) * 2.0 - 1.0;
    vec4 clip = vec4(ndc, -1.0, 1.0);
    vec4 eye = uInvProjection * clip;
    Pcg32State rnd = pcg_srandom(uint(gl_FragCoord.x) +
                uint(gl_FragCoord.y) * 4097U +
                uint(time) * 1234567U);
    eye = vec4(eye.xy, -1.0, 0.0);
    Ray camera_ray = Ray(uCameraPos, normalize((uInvView * eye).xyz));

    float t;
    if (intersectAABB(camera_ray, boxMin, boxMax, t)) {
        vec3 hitPos = camera_ray.origin + (t - 0.01) * camera_ray.direction;
        Ray ray = Ray(hitPos, camera_ray.direction);
        Hit hit = dda(ray, MAX_STEPS);
        if (hit.hit) {
            vec2 palette_coord = vec2(float(voxelmap(hit.voxel)) * inv_palette_size);
            vec3 albedo = texture(u_palette_data, palette_coord).rgb;
            vec3 color = compute_light(camera_ray, hit.position, hit.normal, albedo, rnd);
            fragColor = vec4(color, 1.0);
        } else {
            discard;
        }
    } else {
        discard;
    }
}

#endif
