/**
 * AI Virtual Painter - Toolbar Script
 * Handles tool selection and state management
 */

// Current selected tool
let selectedTool = 'pink';

// Tool display names
const toolNames = {
    pink: 'Pink Brush',
    blue: 'Blue Brush',
    green: 'Green Brush',
    eraser: 'Eraser'
};

/**
 * Select a tool and update the UI
 * @param {string} tool - The tool to select ('pink', 'blue', 'green', or 'eraser')
 */
function selectTool(tool) {
    // Update state
    selectedTool = tool;

    // Remove active class from all buttons
    document.querySelectorAll('.brush-btn, .eraser-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Add active class to selected button
    const selectedBtn = document.querySelector(`[data-color="${tool}"]`);
    if (selectedBtn) {
        selectedBtn.classList.add('active');
    }

    // Update status display
    const statusElement = document.getElementById('selected-tool');
    if (statusElement) {
        statusElement.textContent = toolNames[tool] || tool;
    }

    console.log(`Selected tool: ${tool}`);
}

/**
 * Get the currently selected tool
 * @returns {string} The current tool name
 */
function getSelectedTool() {
    return selectedTool;
}

/**
 * Get the color value for the current tool (for canvas drawing)
 * @returns {string} Hex color code or 'eraser' for the eraser tool
 */
function getToolColor() {
    const colors = {
        pink: '#EC4899',
        blue: '#3B82F6',
        green: '#22C55E',
        eraser: 'eraser'
    };
    return colors[selectedTool] || '#000000';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('AI Virtual Painter Toolbar initialized');
    console.log('Default tool: Pink Brush');

    // Ensure pink is active on load
    selectTool('pink');
});
