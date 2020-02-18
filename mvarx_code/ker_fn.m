function kz = ker_fn(krn_type, corr_window, L)
% KER_FN kernel function

z_idx = (1:corr_window) / L;

switch krn_type
case 'BAR'
    %Bartlett
    kz = 1 - z_idx;
    kz(z_idx > 1) = 0;
case 'DAN'
    %Daniell
    kz = sin(pi * z_idx) ./ (pi * z_idx);
case 'PAR'
    %Parzen
    a = 1 - 6*(pi*z_idx./6).^2 + 6*abs(pi*z_idx./6).^3;
    b = 2 * (1 - abs(pi*z_idx./6)).^3;
    kz = zeros(1, corr_window);
    kz(z_idx <= 3/pi) = a(z_idx <= 3/pi);
    kz(z_idx >= 3/pi & z_idx <= 6/pi) = b(z_idx >= 3/pi & z_idx <= 6/pi);
case 'TR'
    %Truncated Uniform
    kz = zeros(1, corr_window);
    kz(z_idx <= 1) = 1;
case 'QS'
    %Bartlet-Priestly
    kz = 9./(5*pi^2*z_idx.^2).*(sin(pi*sqrt(5/3)*z_idx) ... 
    		./ (pi * sqrt(5/3).*z_idx) - cos(pi*sqrt(5/3)*z_idx));
end