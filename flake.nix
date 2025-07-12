{
  description = "A basic flake for Python Projects";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }: (
    flake-utils.lib.eachDefaultSystem
    (system: let
      pkgs = import nixpkgs {
        inherit system;

        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
    in {
      devShells.default = pkgs.mkShell {
        name = "dev-shell";

        nativeBuildInputs = [];

        packages = with pkgs; [
          (python3.withPackages (python-pkgs:
            with python-pkgs; [
              # python-lsp-server
              # fastapi
              # uvicorn
              # diffusers
              # transformers
              # accelerate
            ]))

            # python312Packages.pytorch-bin
            # python312Packages.diffusers
        ];
      };
    })
  );
}
