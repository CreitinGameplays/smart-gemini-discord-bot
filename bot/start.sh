#!/bin/bash

# Function to run the bot script
run_bot() {
    while true; do
        python v3.5.py 
        if [ $? -eq 1 ]; then
            echo "Bot script exited with error. Restarting..."
            break
        fi
        sleep 1
    done
}

# run the bot
while true; do
    
    # Run the bot script in the background
    run_bot &

    # Wait for the bot script to exit
    wait $!

    sleep 5  # Sleep 
done
