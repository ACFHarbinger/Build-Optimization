mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            commands::data::list_solver_results,
            commands::data::read_solver_result,
            commands::data::list_item_files,
            commands::data::read_items_json,
            commands::training::list_training_runs,
            commands::training::read_training_log,
            commands::tracking::list_experiments,
            commands::tracking::list_tracked_runs,
            commands::tracking::get_run_params,
            commands::tracking::get_run_latest_metrics,
            commands::tracking::get_run_metric_history,
            commands::tracking::get_run_artifacts,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Build-Optimization Studio");
}
