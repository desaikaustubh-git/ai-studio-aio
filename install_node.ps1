# Links this folder's pieces into the Pinokio "minimax-h3-pinokio" ComfyUI so
# its front end can use them. Two directory junctions (no admin, no copies to
# keep in sync, edits here are live). Re-run any time; it just refreshes them.
#
#   1. comfy_nodes\ai_studio_h3_prompt  ->  app\custom_nodes\ai_studio_h3_prompt
#   2. workflows                        ->  app\user\default\workflows\AI Studio
#
# After (1) you must restart ComfyUI. After (2) a browser refresh is enough -
# the workflows show up under  Workflows > AI Studio  and save straight back
# into this folder.
#
# The junction SOURCES are resolved from this script's own location, so moving
# this whole folder just needs a re-run. Only the ComfyUI install path is
# external - override it if yours differs:
#     .\install_node.ps1 -ComfyApp "D:\some\other\ComfyUI\app"

param(
    [string]$ComfyApp = "E:\pinokio\api\minimax-h3-pinokio.git\app"
)

$ErrorActionPreference = "Stop"

$comfyApp = $ComfyApp

function Link-Dir($source, $target) {
    if (-not (Test-Path $source)) { throw "source not found: $source" }
    $parent = Split-Path $target -Parent
    if (-not (Test-Path $parent)) { throw "target parent not found: $parent" }
    if (Test-Path $target) {
        $item = Get-Item $target -Force
        if ($item.LinkType) {
            Write-Host "Removing existing link: $target"
            $item.Delete()
        } else {
            throw "$target exists and is a real folder, not a link. Move it aside first."
        }
    }
    New-Item -ItemType Junction -Path $target -Value $source | Out-Null
    Write-Host "Linked:  $target"
    Write-Host "   ->    $source"
    Write-Host ""
}

Link-Dir (Join-Path $PSScriptRoot "comfy_nodes\ai_studio_h3_prompt") `
         (Join-Path $comfyApp "custom_nodes\ai_studio_h3_prompt")

Link-Dir (Join-Path $PSScriptRoot "workflows") `
         (Join-Path $comfyApp "user\default\workflows\AI Studio")

Write-Host "Now restart ComfyUI (Pinokio > minimax-h3-pinokio > Stop, then Start)"
Write-Host "so it picks up the node, then hard-refresh the browser tab."
Write-Host "The workflows appear under  Workflows > AI Studio."
