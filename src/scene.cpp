#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "tiny_obj_loader.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom{}; // {} so triStart/triCount are 0 for non meshes
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
            std::string sceneDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
            loadMesh(sceneDir + std::string(p["FILE"]), newGeom);
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // depth of field. defaults to pinhole, and the lookat point
    camera.lensRadius = cameraData.value("APERTURE", 0.0f);
    camera.focalDistance = cameraData.value("FOCAL_DIST", glm::length(camera.lookAt - camera.position));

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

/// TINYOBJ LOADING
void Scene::loadMesh(const std::string& meshPath, Geom& geom)
{   
    // remember, this->triangles is shared
    geom.triStart = triangles.size();

    tinyobj::attrib_t attrib;                      // flat arrays of every v / vn / vt in the file
    std::vector<tinyobj::shape_t> shapes;          // each o / g group. faces index into attrib
    std::vector<tinyobj::material_t> objMaterials; // .mtl, unused (material comes from the json)
    std::string warn, err;

    // triangulates splits quads/polygons into triangles, so faces are always 3 indices
    bool ok = tinyobj::LoadObj(&attrib, &shapes, &objMaterials, &warn, &err, meshPath.c_str());
    if (!warn.empty())
    {
        cout << "obj warning (" << meshPath << "): " << warn << endl;
    }
    if (!ok)
    {
        cout << "couldnt load mesh " << meshPath << ": " << err << endl;
        exit(-1);
    }

    // each shape contains mesh indices. collection of triangles
    for (const tinyobj::shape_t& shape : shapes)
    {
        const std::vector<tinyobj::index_t>& idx = shape.mesh.indices;
        // for every triangle (k, k+1, k+2),
        for (size_t k = 0; k + 2 < idx.size(); k += 3)
        {
            glm::vec3 v[3];
            glm::vec3 n[3];
            bool hasNormals = true;
            // for every vertex in that triangle, 
            for (int c = 0; c < 3; c++)
            {
                // each corner has its own position + normal index into attrib (normal is -1 if missing)
                const tinyobj::index_t& id = idx[k + c];

                // fill in the vertex position
                v[c] = glm::vec3(attrib.vertices[3 * id.vertex_index + 0],
                                 attrib.vertices[3 * id.vertex_index + 1],
                                 attrib.vertices[3 * id.vertex_index + 2]);
                if (id.normal_index >= 0)
                {
                    // and normal
                    n[c] = glm::vec3(attrib.normals[3 * id.normal_index + 0],
                                     attrib.normals[3 * id.normal_index + 1],
                                     attrib.normals[3 * id.normal_index + 2]);
                }
                else
                {
                    hasNormals = false;
                }
            }

            if (!hasNormals)
            {
                // no normals in the file, just flat shade with the face normal
                glm::vec3 faceN = glm::cross(v[1] - v[0], v[2] - v[0]);
                float len = glm::length(faceN);
                faceN = len > 0.0f ? faceN / len : glm::vec3(0.0f, 1.0f, 0.0f); // degenerate
                n[0] = faceN; 
                n[1] = faceN; 
                n[2] = faceN;
            }

            // add it into our global vector
            Triangle tri;
            tri.v0 = v[0]; 
            tri.v1 = v[1]; 
            tri.v2 = v[2];
            tri.n0 = glm::normalize(n[0]); 
            tri.n1 = glm::normalize(n[1]); 
            tri.n2 = glm::normalize(n[2]);
            triangles.push_back(tri);
        }
    }

    geom.triCount = triangles.size() - geom.triStart;
    if (geom.triCount == 0)
    {
        cout << "Mesh " << meshPath << " has no triangles" << endl;
        exit(-1);
    }

    // find the bbox over just this mesh's triangles
    glm::vec3 bmin(FLT_MAX);
    glm::vec3 bmax(-FLT_MAX);
    for (int i = geom.triStart; i < geom.triStart + geom.triCount; i++)
    {
        const Triangle& t = triangles[i];
        bmin = glm::min(bmin, glm::min(t.v0, glm::min(t.v1, t.v2)));
        bmax = glm::max(bmax, glm::max(t.v0, glm::max(t.v1, t.v2)));
    }

    // scale the bbox to fit in the unit cube
    glm::vec3 center = 0.5f * (bmin + bmax);
    glm::vec3 rangeBox = bmax - bmin;
    float size = glm::max(rangeBox.x, glm::max(rangeBox.y, rangeBox.z));
    for (int i = geom.triStart; i < geom.triStart + geom.triCount; i++)
    {
        Triangle& t = triangles[i];
        t.v0 = (t.v0 - center) / size;
        t.v1 = (t.v1 - center) / size;
        t.v2 = (t.v2 - center) / size;
    }
    geom.bboxMin = (bmin - center) / size;
    geom.bboxMax = (bmax - center) / size;

    cout << "Loaded mesh " << meshPath << ": " << geom.triCount << " triangles" << endl;
}
