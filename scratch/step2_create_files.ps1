# File paths to create
$files = @(
    "src/simulation/simulator.py",
    "src/preprocessing/processor.py",
    "src/model/net.py",
    "src/model/trainer.py",
    "src/export/gif_writer.py",
    "src/config.py",
    "run.py"
)

# Template content for each Python file
$header = @'
"""
Module: {0}
Description: TODO - Add module purpose here.
"""
'@

foreach ($file in $files) {
    $module = Split-Path $file -Leaf
    $docstring = $header -f $module
    Set-Content -Path $file -Value $docstring -Encoding UTF8
}

Write-Host "Module files created with docstring templates."
