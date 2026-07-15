# line 1 1

float halton(int base, int index) {
    float result = 0.;
    float f = 1.;
    while (index > 0)
    {
        f = f / float(base);
        result += f * float(index % base);
        index = index / base;
    }
    return result;
}

vec2 halton2D(int frame_counter) {
    frame_counter = frame_counter % 32;
    return vec2(halton(2, frame_counter), halton(3, frame_counter));
}

vec3 halton3D(int frame_counter) {
    frame_counter = frame_counter % 32;
    return vec3(halton(2, frame_counter), halton(3, frame_counter), halton(5, frame_counter));
}

vec3 generate_random_normal(inout int seed, sampler2DArray stbn_normals) {
    seed = (seed + 1) % 64;
    vec3 rnd_normal_coord = vec3(mod(gl_FragCoord.xy, 128) / 128.0, seed);
    return texture(stbn_normals, rnd_normal_coord).rgb * 2.0 - 1.0;
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

vec3 generate_random_cosine_weighted_normal(vec3 normal, sampler2DArray stbn_normals, inout int seed) {
    vec3 normal_tangent = generate_random_normal(seed, stbn_normals);
    return tangentToWorld(normal, normal_tangent);
}

vec3 generate_random_stbn_unitvec3(sampler2DArray stbn_normals, inout int seed) {
    seed = (seed + 1) % 64;
    vec3 rnd_normal_coord = vec3(mod(gl_FragCoord.xy, 128) / 128.0, seed);
    return normalize(texture(stbn_normals, rnd_normal_coord).rgb * 2.0 - 1.0);
}
