from mesa.datacollection import DataCollector


def compute_average_performance(model):
    average_perf = model.performance_metrics.calculate_total_performance(model.task_structure.get_task_matrix(),
                                                                         model.realized_matrix)
    return average_perf



