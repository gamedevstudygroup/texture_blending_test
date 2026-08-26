{
  description = "Development shell for the marimo image transport notebook";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python3.withPackages (ps: with ps; [
            marimo
            numpy
            openai
            pillow
            plotly
            pydantic-ai-slim
            pytest
            ruff
            scipy
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [ python ];

            shellHook = ''
              echo "marimo image transport shell"
              echo "Run: marimo edit image_network_simplex_gaussian.py"
            '';
          };
        });
    };
}
