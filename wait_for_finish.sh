#!/bin/bash
while true; do
  if grep -q "Discovery finished" /liuyutian/ai_scientist/AI-DIRAC-strict/run_output2.log 2>/dev/null; then
    echo "RUN COMPLETE at $(date)" >> /liuyutian/ai_scientist/AI-DIRAC-strict/final_results.txt
    tail -80 /liuyutian/ai_scientist/AI-DIRAC-strict/run_output2.log >> /liuyutian/ai_scientist/AI-DIRAC-strict/final_results.txt
    exit 0
  fi
  sleep 300
done
