#!/bin/bash
# Post-training hook - runs after training completes

echo "🎉 Training complete!"

# Get the latest experiment directory
LATEST_RUN=$(ls -t experiments/runs/ | head -1)

if [ ! -z "$LATEST_RUN" ]; then
    echo "📊 Results saved in: experiments/runs/$LATEST_RUN"

    # Show final metrics
    if [ -f "experiments/runs/$LATEST_RUN/metrics.jsonl" ]; then
        echo "📈 Final metrics:"
        tail -1 experiments/runs/$LATEST_RUN/metrics.jsonl | jq '.'
    fi

    # Check if model exists
    if [ -f "experiments/runs/$LATEST_RUN/best_model.pt" ]; then
        echo "✅ Best model saved"
        echo "   Size: $(du -h experiments/runs/$LATEST_RUN/best_model.pt | cut -f1)"
    fi
fi

# Offer to start testing
echo ""
echo "What's next?"
echo "1. Test the model (launches web UI)"
echo "2. Compare with previous runs"
echo "3. Deploy to Hugging Face"
echo "4. Export to ONNX"
echo "5. Nothing, I'll handle it"

# Remove PID file
rm -f .training.pid

# Send notification if configured
if [ ! -z "$NOTIFICATION_WEBHOOK" ]; then
    curl -X POST $NOTIFICATION_WEBHOOK \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"Training complete! Model saved to $LATEST_RUN\"}"
fi