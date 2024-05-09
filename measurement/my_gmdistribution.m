function GMObj = my_gmdistribution(mu, Sigma, p, varargin)
% Create a Gaussian mixture distribution object
% Input arguments:
% mu    - mean of each Gaussian component (k-by-d matrix)
% Sigma - covariance matrix of each Gaussian component (d-by-d-by-k array)
% p     - prior probability of each Gaussian component (1-by-k vector)
% varargin - optional name-value pairs for customization

% Set default values for optional arguments
Regularize = 1e-6;
CovType = 'diagonal';
SharedCov = [];

% Parse optional arguments
if nargin > 3
    for i = 1:2:length(varargin)
        switch varargin{i}
            case 'Regularize'
                Regularize = varargin{i+1};
            case 'CovType'
                CovType = varargin{i+1};
            case 'SharedCov'
                SharedCov = varargin{i+1};
            otherwise
                error(['Unrecognized option: ', varargin{i}]);
        end
    end
end

% Create Gaussian mixture distribution object
GMObj = gmdistribution(mu, Sigma, p, 'Regularize', Regularize, ...
                      'CovType', CovType, 'SharedCov', SharedCov);

