dir = "NYU_Depth_v2_Labeled";
if ~isfolder(dir)
    mkdir(dir)
end

for i = 1:size(images,4)
    file = fullfile(dir,num2str(i) + ".png");
    if ~isfile(file)
        imwrite(images(:,:,:,i), file);
    end
end
