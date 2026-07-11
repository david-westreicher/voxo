# line 0

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

Ray compute_camera_ray(mat4 uInvProjection, mat4 uInvView, vec3 uCameraPos) {
    vec2 ndc = gl_FragCoord.xy / vec2(1920, 1080) * 2.0 - 1.0;
    vec4 clip = vec4(ndc, -1.0, 1.0);
    vec4 eye = uInvProjection * clip;
    eye = vec4(eye.xy, -1.0, 0.0);
    return Ray(uCameraPos, normalize((uInvView * eye).xyz));
}

vec3 skyColor(vec3 rayDir) {
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

uint voxelmap(vec3 p, Box bbox, usampler3D u_voxel_data)
{
    vec3 local_coord = (p - bbox.min + 0.5) / (bbox.max - bbox.min);
    return texture(u_voxel_data, local_coord).r;
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
    for (int i = 0; i < max_steps; i++) {
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
    if (tNear > tFar || tFar < 0.0)
        return false;

    tHit = max(tNear, 0.0);
    return true;
}

Box compute_bbox(sampler3D data_3d) {
    vec3 size = ceil(textureSize(data_3d, 0) * 0.5) * 2.0; // NOTE(david): hack to make odd dimensions work
    vec3 boxMin = -size * 0.5;
    vec3 boxMax = size * 0.5;
    return Box(boxMin, boxMax);
}

Ray transform_to_local_ray(Ray world_ray, mat4 model_inverse) {
    vec3 origin = (model_inverse * vec4(world_ray.origin, 1.0)).xyz;
    vec3 direction = normalize((model_inverse * vec4(world_ray.direction, 0.0)).xyz);
    return Ray(origin, direction);
}
