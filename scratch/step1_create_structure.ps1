$folders = @(
    "notebooks",
    "src/simulation",
    "src/preprocessing",
    "src/model",
    "src/export",
    "tests",
    "data/raw",
    "data/processed",
    "outputs/gifs",
    "outputs/logs"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

# Create __init__.py in every src subfolder
$initFolders = @(
    "src",
    "src/simulation",
    "src/preprocessing",
    "src/model",
    "src/export"
)

foreach ($folder in $initFolders) {
    New-Item -ItemType File -Path "$folder\__init__.py" -Force | Out-Null
}

Write-Host "Folder structure created successfully."
