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
