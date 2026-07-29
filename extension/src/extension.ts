import * as vscode from "vscode";

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.commands.registerCommand("buildOptimization.runOptimization", async () => {
      const terminal = vscode.window.createTerminal("Build Optimization");
      terminal.show();
      terminal.sendText("uv run python main.py");
    }),
    vscode.commands.registerCommand("buildOptimization.validateConfig", async () => {
      vscode.window.showInformationMessage(
        "Build Optimization: config validation is wired up via yamlValidation schemas — see extension/schemas/.",
      );
    }),
  );
}

export function deactivate() {}
