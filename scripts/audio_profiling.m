% FIRST AUDIO PROFILING
% Load data
i = 1;
filename = 'time_profiling/latency_se_20to20_d2_wu5.mat';
data = load(filename);
fields = fieldnames(data);
base_names = unique(regexprep(fields, '_total$|_inference$|_preprocess$', ''));

% Extract configuration (e.g., "20to20") from filename
[~, name_only, ~] = fileparts(filename);
config_match = regexp(name_only, 'latency_se_(.*?)_', 'tokens');
if ~isempty(config_match)
    config_str = config_match{1}{1};
else
    config_str = 'unknown';
end
% Pre-allocate color order (optional, for clarity)
colors = lines(numel(base_names));


% Plot TOTAL latencies
figure; hold on; grid on;
name = base_names{i};
total = data.([name '_total']);
plot(total, 'DisplayName', strrep(name, '_', '\_'), 'Color', colors(i,:));
title('Total Latency per Frame');
title(['Mean Total Latency for "' strrep(name, '_', '\_') '"- ' config_str]);
xlabel('Frame Index'); ylabel('Latency (ms)');
ylim([0 20]);
legend show;

% Plot INFERENCE latencies
figure; hold on; grid on;
name = base_names{i};
inf = data.([name '_inference']);
plot(inf, 'DisplayName', strrep(name, '_', '\_'), 'Color', colors(i,:));
title(['Inference Latency per Frame - ' config_str]);
xlabel('Frame Index'); ylabel('Latency (ms)');
ylim([0 20]);
legend show;


% Plot PREPROCESS latencies
figure; hold on; grid on;
name = base_names{i};
pre = data.([name '_preprocess']);
plot(pre, 'DisplayName', strrep(name, '_', '\_'), 'Color', colors(i,:));
title(['Preprocess Latency per Frame - ' config_str]);
xlabel('Frame Index'); ylabel('Latency (ms)');
legend show;

