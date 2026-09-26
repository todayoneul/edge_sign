# Logs the busiest processes every 5 s while a measurement runs (share of all logical CPUs).
# Usage: powershell -File cpu_sampler.ps1 -Out <log> -StopFile <path>   (stops when StopFile exists)
param([string]$Out, [string]$StopFile)
$cores = [Environment]::ProcessorCount
while (-not (Test-Path $StopFile)) {
  $s = Get-Counter '\Process(*)\% Processor Time' -SampleInterval 5 -MaxSamples 1 -ErrorAction SilentlyContinue
  if ($s) {
    $top = $s.CounterSamples | Where-Object { $_.InstanceName -notin @('_total', 'idle') } |
      Group-Object InstanceName | ForEach-Object { [pscustomobject]@{ n = $_.Name; v = ($_.Group | Measure-Object CookedValue -Sum).Sum / $cores } } |
      Sort-Object v -Descending | Select-Object -First 6
    $line = (Get-Date -Format 'HH:mm:ss') + '  ' + (($top | ForEach-Object { '{0}={1:N1}' -f $_.n, $_.v }) -join '  ')
    Add-Content -Path $Out -Value $line -Encoding utf8
  }
}
