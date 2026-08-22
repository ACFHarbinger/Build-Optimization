// BuildOptimization module implementation.
//
// Provides the module startup/shutdown hooks. For the U1 skeleton these are
// empty; the intended later work is registration of the in-editor Slate panel
// (U3, in a separate Editor module) and Blueprint-callable nodes (U4).

#include "BuildOptimizationModule.h"

#define LOCTEXT_NAMESPACE "FBuildOptimizationModule"

void FBuildOptimizationModule::StartupModule()
{
	// Nothing to register yet. U3/U4 will register the editor panel and
	// Blueprint nodes here (or in the Editor module's StartupModule).
}

void FBuildOptimizationModule::ShutdownModule()
{
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FBuildOptimizationModule, BuildOptimization)
