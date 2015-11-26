function [ouput_matrix] = im_padding (varargin)

InputMatrix = varargin{1};
PaddingSize = varargin{2};
PaddingColor = varargin{3};

[x, y] = size(InputMatrix);
PaddedMatrix = zeros(x+PaddingSize*2, y+PaddingSize*2);
PaddedMatrix(:,:) =  PaddingColor;
PaddedMatrix(PaddingSize+1:PaddingSize+x, PaddingSize+1:PaddingSize+y) = InputMatrix;

ouput_matrix = PaddedMatrix;

end
