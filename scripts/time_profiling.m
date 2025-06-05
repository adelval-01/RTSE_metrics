%% ALL AUDIOS PROFILING
% Load the .mat file
data = load('all_audio_latencies.mat');
whos

% Get all field names in the struct
fields = fieldnames(data);

fields{0}

% Extract unique audio file names (strip "_total", "_inference", "_preprocess")
% base_names = unique(regexprep(fields, '_total$|_inference$|_preprocess$', ''));


% Identify latency types
latency_types = {'total', 'inference', 'preprocess'};

for lt = 1:length(latency_types)
    latency_type = latency_types{lt};

    % Create figure for this latency type
    figure;
    hold on;
    
    % Filter fields for this latency type
    matching_fields = fields(endsWith(fields, ['_' latency_type]));

    for i = 1:length(matching_fields)
        var_name = matching_fields{i};
        values = data.(var_name);

        % Clean the name for legend
        audio_name = erase(var_name, ['_' latency_type]);
        audio_name = strrep(audio_name, '_', '\_');

        plot(values, 'DisplayName', audio_name);
    end

    title([upper(latency_type(1)) latency_type(2:end) ' Latency per Frame']);
    xlabel('Frame Index');
    ylabel('Latency (ms)');
    legend('show');
    grid on;
end
