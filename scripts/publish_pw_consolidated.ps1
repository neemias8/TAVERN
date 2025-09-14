param(
  [Parameter(Mandatory=$true)] [string]$GithubToken,
  [string]$Owner = "neemias8",
  [string]$RepoName = "PW_CONSOLIDATED",
  [switch]$Private,
  [string]$Description = "Consolidated Gospel harmony (integral single-source; longest multi-source)",
  [string]$Homepage = ""
)

function Invoke-GitHubApi {
  param(
    [string]$Method,
    [string]$Uri,
    [object]$Body = $null
  )
  $headers = @{
    Authorization = "token $GithubToken"
    Accept        = "application/vnd.github+json"
    'X-GitHub-Api-Version' = '2022-11-28'
    'User-Agent' = 'pw-consolidated-script'
  }
  if ($Body -ne $null) {
    $json = ($Body | ConvertTo-Json -Depth 10)
    return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $headers -ContentType 'application/json' -Body $json
  }
  else {
    return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $headers
  }
}

Write-Host "Checking if repo '$Owner/$RepoName' exists..."
$repoUri = "https://api.github.com/repos/$Owner/$RepoName"
$exists = $false
try {
  $null = Invoke-GitHubApi -Method GET -Uri $repoUri
  $exists = $true
  Write-Host "Repo already exists."
} catch {
  if ($_.Exception.Response.StatusCode.Value__ -eq 404) {
    $exists = $false
  } else {
    throw $_
  }
}

if (-not $exists) {
  Write-Host "Creating repo '$Owner/$RepoName'..."
  # Create under user account
  $create = Invoke-GitHubApi -Method POST -Uri 'https://api.github.com/user/repos' -Body @{
    name        = $RepoName
    description = $Description
    homepage    = $Homepage
    @private    = [bool]$Private
  }
  Write-Host "Created: $($create.full_name)"
}

# Add remote if missing
$remoteName = 'pw'
$remoteUrl  = "https://github.com/$Owner/$RepoName.git"
$remotes = git remote
if (-not ($remotes -split "\r?\n" | Where-Object { $_ -eq $remoteName })) {
  git remote add $remoteName $remoteUrl
  Write-Host "Added remote '$remoteName' -> $remoteUrl"
} else {
  git remote set-url $remoteName $remoteUrl
  Write-Host "Updated remote '$remoteName' -> $remoteUrl"
}

# Push current branch (main) to the new remote
Write-Host "Pushing 'main' to '$remoteName'..."
git push -u $remoteName main
Write-Host "Done."

