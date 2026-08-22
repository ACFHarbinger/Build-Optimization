// BuildOptimization module public interface.
//
// Standard UE module entry point: a private IModuleInterface subclass whose
// StartupModule/ShutdownModule hooks let the module register/defer any global
// resources (e.g. editor extensions, Blueprint nodes, or a singleton service
// facade for the C++ solver core later). The skeleton is deliberately empty
// beyond those hooks.
//
// This header is the module's Public root. Future public-facing types (the
// Item/Build/Synergy domain proxies that U2 marshals) belong under
// Public/ next to it.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FBuildOptimizationModule : public IModuleInterface
{
public:
	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
