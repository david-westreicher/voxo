#include programs/random.glsl
# line 2 2

#define SCREEN_DIMENSIONS vec2(1, 1)

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

struct Box {
    vec3 min;
    vec3 max;
};

Ray compute_camera_ray(vec2 screen_uv, mat4 uInvProjection, mat4 uInvView, int frame_counter, float jitter_scale) {
    vec2 jitter = halton2D(frame_counter) - vec2(0.5);
    vec2 ndc = (screen_uv + jitter * jitter_scale / SCREEN_DIMENSIONS) * 2.0 - 1.0;
    vec4 clip = vec4(ndc, -1.0, 1.0);
    vec4 eye = uInvProjection * clip;
    eye = vec4(eye.xy, -1.0, 0.0);
    vec3 cameraPos = uInvView[3].xyz;
    return Ray(cameraPos, normalize((uInvView * eye).xyz));
}

vec3 skyColor(vec3 rd, vec3 sunDir)
{
    float up = max(rd.y, 0.0);

    vec3 zenith = vec3(0.18, 0.35, 1.00);
    vec3 horizon = vec3(0.75, 0.85, 1.20);
    vec3 ground = vec3(0.03);

    vec3 color = mix(horizon, zenith, pow(up, 0.35));

    color = mix(
            ground,
            color,
            smoothstep(-0.05, 0.02, rd.y)
        );

    // Warm horizon glow
    color += vec3(1.0, 0.6, 0.2) * pow(1.0 - up, 6.0) * 0.5;

    if (sunDir.y >= 0) {
        float sun = max(dot(rd, sunDir), 0.0);
        // Sun halo
        color += vec3(20.0, 18.0, 14.0) * pow(sun, 128.0);

        // Sun disc (HDR)
        color += vec3(200.0) * pow(sun, 40.0);
    }
    return color;
}

vec3 skyColor2(vec3 rayDir) {
    vec3 dir = normalize(rayDir);

    // Map y from [-1,1] to [0,1]
    float t = 0.5 * (dir.y + 1.0);

    vec3 horizon = vec3(0.8, 0.9, 1.0);
    vec3 zenith = vec3(0.2, 0.4, 0.8);

    return mix(horizon, zenith, t) * 2.5;
}

bool is_inside_box(vec3 p, Box box) {
    return all(greaterThanEqual(p, box.min)) &&
        all(lessThan(p, box.max));
}

uint voxelmap(vec3 p, Box bbox, usampler3D u_voxel_data) {
    vec3 local_coord = (p + 0.5) / (bbox.max - bbox.min);
    return textureLod(u_voxel_data, local_coord, 0.0).r;
}

Hit dda(Ray ray, int max_steps, usampler3D voxels, Box bbox) {
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

    bool has_entered = is_inside_box(map, bbox);
    int i;
    for (i = 0; i < max_steps; i++) {
        vec4 conds = step(sideDist.xxyy, sideDist.yzzx);
        vec3 cases = vec3(0);
        cases.x = conds.x * conds.y;
        cases.y = (1. - cases.x) * conds.z * conds.w;
        cases.z = (1. - cases.x) * (1. - cases.y);
        sideDist += max((2. * cases - 1.) * deltaDist, 0.);
        map += cases * stepDir;
        if (is_inside_box(map, bbox) && !has_entered) {
            has_entered = true;
        }
        if (has_entered && !is_inside_box(map, bbox)) {
            return hit;
        }
        if (has_entered && voxelmap(map, bbox, voxels) > 0.) // Did we hit anything? if so, we are done!
        {
            side = cases.y + 2. * cases.z;
            break;
        }
    }
    if (!has_entered || i == max_steps) {
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

vec2 world_to_uv(vec3 world_pos, mat4x4 projectionview) {
    vec4 clip = projectionview * vec4(world_pos, 1.0);
    if (clip.w <= 0.0)
        return vec2(-1.0); // behind the camera
    vec3 ndc = clip.xyz / clip.w;
    return ndc.xy * 0.5 + 0.5;
}

Hit screen_space_dda(Ray ray, int max_steps, usampler3D voxels, mat4x4 projview, sampler2D linear_depth, vec3 camera_pos, Box bbox) {
    vec3 world_pos = ray.origin + ray.direction;
    vec2 uv = world_to_uv(world_pos, projview);
    float screen_depth = texture(linear_depth, uv).r;
    float sample_depth = distance(world_pos, camera_pos);
    if (all(greaterThanEqual(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)))
            && screen_depth < sample_depth && sample_depth - screen_depth < 1.5) {
        Hit hit;
        hit.hit = true;
        hit.t = distance(ray.origin, world_pos);
        return hit;
    }
    ray.origin = world_pos;
    return dda(ray, max_steps, voxels, bbox);
}

bool intersectAABB(
    Ray ray,
    Box bbox,
    out float tHit
) {
    vec3 invDir = 1.0 / ray.direction;

    vec3 t0 = (bbox.min - ray.origin) * invDir;
    vec3 t1 = (bbox.max - ray.origin) * invDir;

    vec3 tMin = min(t0, t1);
    vec3 tMax = max(t0, t1);

    float tNear = max(max(tMin.x, tMin.y), tMin.z);
    float tFar = min(min(tMax.x, tMax.y), tMax.z);

    // No intersection, or box is behind ray
    if (tNear > tFar || tFar < 0.0) {
        discard;
        return false;
    }

    tHit = max(tNear, 0.0);
    return true;
}

Box compute_bbox(usampler3D data_3d) {
    ivec3 size = textureSize(data_3d, 0);
    return Box(vec3(0.0), vec3(size));
}

Ray transform_to_local_ray(Ray world_ray, mat4 model_inverse) {
    vec3 origin = (model_inverse * vec4(world_ray.origin, 1.0)).xyz;
    vec3 direction = normalize((model_inverse * vec4(world_ray.direction, 0.0)).xyz);
    return Ray(origin, direction);
}

vec3 decodeNormalRGB10A2(vec3 encoded)
{
    // map [0,1] -> [-1,1]
    vec3 decoded = encoded * 2.0 - 1.0;
    return normalize(decoded);
}
