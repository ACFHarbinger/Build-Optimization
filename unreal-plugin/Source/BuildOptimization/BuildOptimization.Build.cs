// Copyright Epic Games-style build rules for the Build Optimization module.
//
// Follows standard UE module conventions: one rules file per module, derived
// from ModuleRules, declaring Public/Private dependency modules. This runtime
// module is intentionally minimal for the U1 skeleton -- U3 (Slate editor
// panel) will add an Editor module that depends on this one, and U5 (C++
// solver bridge) will add the real solver dependency here.
//
// Note: this file is authored against UE5 module conventions but was NOT
// compiled (no Unreal Engine install in this environment). See the bus
// landing note for how it was validated.

using UnrealBuildTool;

public class BuildOptimization : ModuleRules
{
	public BuildOptimization(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		// Runtime dependency surface is intentionally empty for the skeleton.
		// Public: module headers other modules may include.
		// Private: implementation deps (none yet).
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				// The C++ solver core (backend/) is bridged here in U5; it is
				// intentionally not referenced yet so the plugin authors against
				// a stock UE project without pulling the backend toolchain.
			}
		);
	}
}
