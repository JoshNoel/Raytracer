#include "Scene.h"
#include "Node.h"

Scene::Scene()
{
    SQRT_DIV2_SHADOW_SAMPLES = std::sqrt(SHADOW_SAMPLES) / 2;
}

Scene::~Scene()
{

}