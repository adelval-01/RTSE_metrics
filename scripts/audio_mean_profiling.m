%% HISTOGRAM
% Load data
file = 'time_profiling/latency_se_20to20_d4_wu5.mat'
data = load(file);
fields = fieldnames(data);

% Use regular expression to extract digit(s) after "wu"
% tokens = regexp(file, 'wu(\d+)', 'tokens');
tokens = regexp(file, 'd(\d+)', 'tokens');

if ~isempty(tokens)
    wu_value = str2double(tokens{1}{1});
else
    wu_value = NaN; % Or handle it another way if needed
end

% Extract base audio names
base_names = unique(regexprep(fields, '_total$|_inference$|_preprocess$', ''));

% Compute mean total latency for each audio
mean_totals = zeros(length(base_names), 1);
for i = 1:length(base_names)
    name = base_names{i};
    total = data.([name '_total']);
    mean_totals(i) = mean(total);
end

% Create bar plot
figure;
bar(mean_totals);
set(gca, 'XTickLabel', strrep(base_names, '_', '\_'), 'XTickLabelRotation', 45);
ylabel('Mean Total Latency (ms)');

title(sprintf('Mean Latency per Frame [diezmation factor = %d]', wu_value));
grid on;
