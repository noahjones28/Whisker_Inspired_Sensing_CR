%% Estimate applied force from known proximal-end load cell values
function [Fx_estimate, MB_estimate, MD_estimate, F_estimate, s_estimate, z_estimate] = force_estimation_from_cell(n, x, y, z, E, I, G, J, Fx_cell, MB_cell, MD_cell)
    % use force estimation script to calculate remaining values
    [Fx_estimate, MB_estimate, MD_estimate, F_estimate, s_estimate, z_estimate] = force_estimation_script(n, x, y, z, E, I, G, J, Fx_cell, MB_cell, MD_cell);
end