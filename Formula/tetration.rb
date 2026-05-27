# Documentation: https://docs.brew.sh/Formula-Cookbook
class Tetration < Formula
  desc "Mmap-oriented chunked tensor format and tet CLI (query, convert, info)"
  homepage "https://github.com/thicclatka/tetration"
  url "https://github.com/thicclatka/tetration/archive/refs/tags/v0.1.5.tar.gz"
  sha256 "6ae0dc8e12376140d9f0342f3da8804469c828bbcd8ca9518ca3558343f825a8"
  license any_of: ["MIT", "Apache-2.0"]

  depends_on "pkgconf" => :build
  depends_on "rust" => :build

  depends_on "hdf5"
  depends_on "netcdf"

  def install
    hdf5 = Formula["hdf5"].opt_prefix
    netcdf = Formula["netcdf"].opt_prefix
    ENV["HDF5_DIR"] = hdf5
    ENV["HDF5_ROOT"] = hdf5
    ENV["HDF5_INCLUDE_DIR"] = "#{hdf5}/include"
    ENV["HDF5_LIB_DIR"] = "#{hdf5}/lib"
    ENV["NETCDF_DIR"] = netcdf
    ENV.prepend_path "PKG_CONFIG_PATH", "#{hdf5}/lib/pkgconfig"
    ENV.prepend_path "PKG_CONFIG_PATH", "#{netcdf}/lib/pkgconfig"
    system "cargo", "install", *std_cargo_args
  end

  test do
    assert_match "Tetration CLI", shell_output("#{bin}/tet --help")
  end
end
