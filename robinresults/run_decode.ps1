# ROBIN detect-only on existing 1K images (uses saved neg_scores, resumes per attack)
$py = "D:\waves\.venv\Scripts\python.exe"
$bench = "D:\waves\ROBIN\mainbenchmark"

Set-Location D:\waves\ROBIN
& $py $bench `
  --detect-only `
  --image-dir D:\waves\robinresults\images `
  --neg-scores D:\waves\robinresults\neg_scores.npy `
  --output D:\waves\robinresults\benchmark `
  --resume

Write-Host "`nDecode table:"
& $py D:\waves\ROBIN\decode_results.py D:\waves\robinresults\benchmark
