clc; clear all; close all;

path = 'T:/home/wei/ML/Project/MDetect-project/dec_output/';
folder_list = dir(path);

lable_path = 'T:/data/wei/dataset/MDetection/ICPR2012/testing_data/scanner_A/label_cent/';

export_dir = 'T:/home/wei/ML/Project/MDetect-project/export_results/';
if (~exist(export_dir, 'dir'))
    mkdir(export_dir);
end

Total_TP = 0;
Total_FP = 0;
Total_FN = 0;

patch_mapping_table = build_patch_mapping_table();

% Calculate the coordinates
for f_idx=3:length(folder_list)
    
    fprintf([folder_list(f_idx).name '\n'])
    
    % Read label data
    label = csvread([lable_path folder_list(f_idx).name '.csv']);
    
    % Search for all the patch images
    curr_inputs_path = [path folder_list(f_idx).name '/mask/'];
    image_list = dir(curr_inputs_path);        
    
    output_table = {32, 4};
    count = 1;
    for i_idx=3:length(image_list)

        curr_pacth_order = split(image_list(i_idx).name, ["_", "."]);
        curr_pacth_order = str2num(curr_pacth_order{4}) + 1;
        
        curr_image = double(imread([curr_inputs_path image_list(i_idx).name]));
        
        vote_num = sum(sum(curr_image));
        
        if(vote_num > 100)
            
            if curr_pacth_order == 25 && strcmp(folder_list(f_idx).name, 'A01_06')
                curr_pacth_order = curr_pacth_order;
            end
            
%             [h, w] = find(curr_image);
%             acc_h = double(0);
%             acc_w = double(0);
%             for c_idx=1:length(h)
%                 acc_h = acc_h + (h(c_idx)-1) * curr_image(h(c_idx),w(c_idx));
%                 acc_w = acc_w + (w(c_idx)-1) * curr_image(h(c_idx),w(c_idx));
%             end
% 
%             output_w = round((acc_w / vote_num) + patch_mapping_table(curr_pacth_order, 2));
%             output_h = round((acc_h / vote_num) + patch_mapping_table(curr_pacth_order, 1));
            
            c_table = find_corrdinate(curr_image);
            if(vote_num > 450)
                fprintf("[%s] vote_num = %d\n", image_list(i_idx).name, vote_num);
                seed_num = 2;
            else
                seed_num = 1;
            end                     
            
            [~, center] = kmeans(c_table, seed_num);
            
            [c_num, ~] = size(center);
            for c_idx=1:c_num
                
                output_h = round(center(c_idx, 1) + patch_mapping_table(curr_pacth_order, 1));
                output_w = round(center(c_idx, 2) + patch_mapping_table(curr_pacth_order, 2));
                
                if curr_pacth_order >= 65 && curr_pacth_order <= 72
                    if output_h > 2047 && output_w < (2083-256)
                        output_table = export_to_table(output_table, image_list(i_idx).name, output_h, output_w, count);
                        count = count + 1;
                    end
                elseif curr_pacth_order >= 73 && curr_pacth_order <= 80
                    if output_w > 2047 && output_h < (2083-256)
                        output_table = export_to_table(output_table, image_list(i_idx).name, output_h, output_w, count);
                        count = count + 1;
                    end                
                elseif curr_pacth_order == 81
                    if output_h > 2047 && output_w > 2047
                        output_table = export_to_table(output_table, image_list(i_idx).name, output_h, output_w, count);
                        count = count + 1;         
                    end
                else
                    output_table = export_to_table(output_table, image_list(i_idx).name, output_h, output_w, count);
                    count = count + 1;
                end           
            end
        end
    end   
    if (count-1) == 0
        output_table = export_to_table(output_table, "None", -1, -1, count);
    end
    
    % Checking the accuracy
    [TP_count, FP_count, FN_count, output_table] = check_accuracy(label, output_table);
    Total_TP = Total_TP + TP_count;
    Total_FP = Total_FP + FP_count;
    Total_FN = Total_FN + FN_count;
    
    export_to_csv(output_table, export_dir, folder_list(f_idx).name);
    
end

recall = Total_TP / (Total_TP + Total_FN);
precision = Total_TP / (Total_TP + Total_FP);
F1 = 2 * (precision * recall)/(precision + recall);

fprintf('Recall: [%f]\n', recall);
fprintf('Precision: [%f]\n', precision);
fprintf('F1: [%f]\n', F1);
fprintf('Done.')

% =========================================================================
% Sub-functions ===========================================================
% =========================================================================

function patch_mapping_table = build_patch_mapping_table()
    % Build the mapping table
    patch_mapping_table = zeros(81, 2); % (:, 1) for h, (:, 2) for w
    for h_idx=0:7
        for w_idx=0:7
            patch_mapping_table(h_idx*8+w_idx+1, 1) = 0 + h_idx*256;
            patch_mapping_table(h_idx*8+w_idx+1, 2) = 0 + w_idx*256;
        end    
    end
    h_idx = 8;
    for w_idx=0:7
        patch_mapping_table(65+w_idx, 1) = (2083-256);
        patch_mapping_table(65+w_idx, 2) = w_idx*256;    
    end
    w_idx = 8;
    for h_idx=0:7
        patch_mapping_table(73+h_idx, 1) = h_idx*256;
        patch_mapping_table(73+h_idx, 2) = (2083-256);    
    end
    patch_mapping_table(81, 1) = (2083-256);
    patch_mapping_table(81, 2) = (2083-256);

end

function table = find_corrdinate(image)
    
    table = zeros(sum(sum(image)),2);
    [h, w] = find(image);
    start_idx = 1;
    
    for i_idx=1:length(h)
        repeat_times = image(h(i_idx), w(i_idx));
        table(start_idx:start_idx+repeat_times-1, 1) = h(i_idx);
        table(start_idx:start_idx+repeat_times-1, 2) = w(i_idx);
        start_idx = start_idx + repeat_times;
    end
    
end

function table = export_to_table(table, file_name, h, w, count)

    table{count, 1} = file_name;
    table{count, 2} = w;
    table{count, 3} = h;
    table{count, 4} = 'None';    
   
end

function [TP_count, FP_count, FN_count, table] = check_accuracy(label, table)
    
    TP_count = 0;
    FP_count = 0;

    [pred_rows, ~] = size(table);    
    [label_rows, ~] = size(label);
    
    if strcmp(table{1, 1}, 'None')
        FN_count = label_rows;
        return;
    end
    
    for p_idx=1:pred_rows
        
        if label_rows == 0
            break;
        end
        
        curr_w = table{p_idx, 2};
        curr_h = table{p_idx, 3};
        
        for r_idx=1:label_rows
            
            curr_dist = ((curr_w-label(r_idx, 1))^2 + (curr_h-label(r_idx, 2))^2)^0.5;
            
            if curr_dist <= 50
                table{p_idx, 4} = 'TP';
                TP_count = TP_count + 1;
                label(r_idx, :) = [];
                [label_rows, ~] = size(label); % Update the size
                break;
            end
        end
        
    end
    
    for p_idx=1:pred_rows
        if ~strcmp(table{p_idx, 4}, 'TP')
            table{p_idx, 4} = 'FP';
            FP_count = FP_count + 1;
        end
    end
    
    FN_count = label_rows;
    
end

function export_to_csv(table, path, file_name)

    fid = fopen([path file_name '.csv'], 'w');
    [rows, ~] = size(table);
    for r_idx=1:rows
        fprintf(fid, '%s, %d, %d, %s\n', table{r_idx, 1}, table{r_idx, 2}, table{r_idx, 3}, table{r_idx, 4});
    end
    fclose(fid);
    
end
