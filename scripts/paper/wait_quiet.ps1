# Blocks until no background process uses >= 3% of all logical CPUs for 3 consecutive 3-second
# samples (processes that belong to the measurement itself are ignored). Gives up after 20 min.
$cores = [Environment]::ProcessorCount
$ignore = @('_total', 'idle', 'python', 'chrome', 'powershell', 'system', 'bash', 'curl')
$streak = 0
$deadline = (Get-Date).AddMinutes(20)
while ($streak -lt 3 -and (Get-Date) -lt $deadline) {
  try {
    $s = Get-Counter '\Process(*)\% Processor Time' -SampleInterval 3 -MaxSamples 1 -ErrorAction Stop
    $top = $s.CounterSamples | Where-Object { $ignore -notcontains $_.InstanceName } |
      Group-Object InstanceName | ForEach-Object { [pscustomobject]@{ n = $_.Name; v = ($_.Group | Measure-Object CookedValue -Sum).Sum / $cores } } |
      Sort-Object v -Descending | Select-Object -First 1
    if ($top.v -lt 3) { $streak++ } else { $streak = 0; Write-Output ("busy: {0} {1:N1}%" -f $top.n, $top.v) }
  } catch { $streak = 0 }
}
if ($streak -ge 3) { Write-Output "quiet" } else { Write-Output "timeout (still busy)" }
