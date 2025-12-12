#!/usr/bin/env python3
#!/usr/bin/env python3
# Terminal-Based Trixel Composer: Autonomous AI Artist (No GUI Dependencies)
# Runs immediately with just the Python 3 standard library (Pillow optional for PNG export)

import os
import asyncio
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from enum import Enum

# --- Configuration ---
CANVAS_WIDTH = 16
CANVAS_HEIGHT = 16
ZW_SESSION_PATH = ".zw/trixel.session"

class CreativePhase(Enum):
    PLANNING = "planning"
    ACTIVE_CREATION = "active_creation"
    REFLECTION = "reflection"
    STYLE_DEVELOPMENT = "style_development"

@dataclass
class CreativeAction:
    tool: str
    x: int
    y: int
    color: Tuple[int, int, int]
    pressure: float = 1.0
    reasoning: str = ""
    timestamp: float = 0.0
    artistic_success: float = 0.5

class TerminalCanvas:
    """ASCII art representation of canvas for terminal display"""

    def __init__(self, width=16, height=16):
        self.width = width
        self.height = height
        self.canvas = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]

    def set_pixel(self, x, y, color):
        """Set pixel color"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.canvas[y][x] = list(color)

    def _pixel_sum(self, pixel):
        return sum(pixel)

    def display(self):
        """Display canvas as ASCII art with color indicators"""
        print("\n" + "="*50)
        print("🎨 TRIXEL COMPOSER - AUTONOMOUS AI ARTIST")
        print("="*50)

        # Create ASCII representation
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                pixel_sum = self._pixel_sum(self.canvas[y][x])
                if pixel_sum == 0:  # Black/empty
                    row += "  "
                elif pixel_sum < 150:  # Dark colors
                    row += "▓▓"
                elif pixel_sum < 400:  # Medium colors
                    row += "▒▒"
                else:  # Light colors
                    row += "░░"
            print(f"|{row}|")

        print("="*50)

    def get_stats(self):
        """Get canvas statistics"""
        non_zero_pixels = sum(
            1 for row in self.canvas for pixel in row if self._pixel_sum(pixel) != 0
        )
        total_pixels = self.width * self.height
        completion = non_zero_pixels / total_pixels

        unique_colors = {tuple(pixel) for row in self.canvas for pixel in row}
        unique_colors.discard((0, 0, 0))

        return {
            'completion': completion,
            'colors_used': len(unique_colors),
            'pixels_painted': non_zero_pixels
        }

class AutonomousCreativeMemory:
    """Simplified memory system for autonomous learning"""
    
    def __init__(self):
        self.short_term = []
        self.tool_preferences = {}
        self.style_evolution = []
        self.session_learnings = []
        
    def add_experience(self, action: CreativeAction, quality: float):
        """Add creative experience to memory"""
        experience = {
            'action': asdict(action),
            'quality': quality,
            'timestamp': time.time()
        }
        
        self.short_term.append(experience)
        if len(self.short_term) > 10:
            self.short_term.pop(0)
            
        # Update tool preferences
        tool = action.tool
        if tool not in self.tool_preferences:
            self.tool_preferences[tool] = {'usage': 0, 'avg_quality': 0.5}
            
        prefs = self.tool_preferences[tool]
        prefs['usage'] += 1
        
        # Moving average for quality
        alpha = 0.2
        prefs['avg_quality'] = (1 - alpha) * prefs['avg_quality'] + alpha * quality
        
    def get_preferred_tool(self):
        """Get currently preferred tool based on success"""
        if not self.tool_preferences:
            return 'brush'
            
        best_tool = max(self.tool_preferences.keys(), 
                       key=lambda t: self.tool_preferences[t]['avg_quality'])
        return best_tool
        
    def get_memory_summary(self):
        """Get summary of current memory state"""
        return {
            'experiences': len(self.short_term),
            'tools_learned': len(self.tool_preferences),
            'preferred_tool': self.get_preferred_tool(),
            'tool_mastery': {tool: f"{data['avg_quality']:.2f}" 
                           for tool, data in self.tool_preferences.items()}
        }

class TerminalTrixelComposer:
    """Terminal-based autonomous AI artist"""
    
    def __init__(self):
        self.canvas = TerminalCanvas()
        self.memory = AutonomousCreativeMemory()
        self.creative_phase = CreativePhase.PLANNING
        self.session_id = f"terminal_{int(time.time())}"
        
        # Color palettes for different phases
        self.phase_colors = {
            'planning': [(139, 69, 19), (160, 82, 45), (205, 133, 63)],
            'active': [(255, 140, 0), (255, 69, 0), (255, 215, 0)],
            'reflection': [(100, 149, 237), (123, 104, 238), (147, 112, 219)],
            'style': [(255, 20, 147), (255, 105, 180), (255, 182, 193)]
        }
        
    def perceive(self):
        """Analyze current canvas state"""
        stats = self.canvas.get_stats()
        if hasattr(self.memory, "get_memory_summary"):
            memory_summary = self.memory.get_memory_summary()
        else:
            experiences = len(getattr(self.memory, "short_term", []))
            tools = getattr(self.memory, "tool_preferences", {})
            memory_summary = {
                'experiences': experiences,
                'tools_learned': len(tools),
                'preferred_tool': getattr(self.memory, "get_preferred_tool", lambda: "brush")(),
                'tool_mastery': {tool: "unknown" for tool in tools}
            }

        return {
            'canvas_stats': stats,
            'memory_state': memory_summary,
            'phase': self.creative_phase.value,
            'session_id': self.session_id
        }
        
    def plan_action(self, perception):
        """Plan next creative action based on phase and memory"""
        stats = perception['canvas_stats']
        memory = perception['memory_state']
        
        # Choose location
        if self.creative_phase == CreativePhase.PLANNING:
            # Start near center
            x = CANVAS_WIDTH // 2 + random.randint(-2, 2)
            y = CANVAS_HEIGHT // 2 + random.randint(-2, 2)
        else:
            # Random exploration with some bias toward existing work
            if stats['completion'] > 0.3:
                # Bias toward existing pixels
                x = random.randint(max(0, CANVAS_WIDTH//2 - 4), 
                                 min(CANVAS_WIDTH-1, CANVAS_WIDTH//2 + 4))
                y = random.randint(max(0, CANVAS_HEIGHT//2 - 4), 
                                 min(CANVAS_HEIGHT-1, CANVAS_HEIGHT//2 + 4))
            else:
                x = random.randint(0, CANVAS_WIDTH - 1)
                y = random.randint(0, CANVAS_HEIGHT - 1)
        
        # Choose tool based on memory or default
        tool = memory.get('preferred_tool', 'brush')
        
        # Choose color based on phase
        phase_key = {
            CreativePhase.PLANNING: 'planning',
            CreativePhase.ACTIVE_CREATION: 'active',
            CreativePhase.REFLECTION: 'reflection',
            CreativePhase.STYLE_DEVELOPMENT: 'style'
        }.get(self.creative_phase, 'active')
        
        colors = self.phase_colors[phase_key]
        color = colors[random.randint(0, len(colors) - 1)]
        
        return CreativeAction(
            tool=tool,
            x=x,
            y=y,
            color=color,
            pressure=random.uniform(0.5, 1.0),
            reasoning=f"{self.creative_phase.value}_action",
            timestamp=time.time()
        )
        
    def execute_action(self, action: CreativeAction):
        """Execute creative action and assess quality"""
        self.canvas.set_pixel(action.x, action.y, action.color)
        
        # Simple quality assessment
        stats = self.canvas.get_stats()
        
        # Quality factors
        base_quality = 0.5
        
        # Reward progress
        if stats['completion'] > 0.1:
            base_quality += 0.2
            
        # Reward color diversity
        if stats['colors_used'] > 3:
            base_quality += 0.1
            
        # Add some randomness for experimentation
        quality = base_quality + random.uniform(-0.2, 0.3)
        quality = max(0.0, min(1.0, quality))
        
        return quality
        
    def update_phase(self, perception):
        """Update creative phase based on progress"""
        completion = perception['canvas_stats']['completion']
        experiences = len(self.memory.short_term)
        
        if completion < 0.2:
            self.creative_phase = CreativePhase.PLANNING
        elif completion < 0.6:
            self.creative_phase = CreativePhase.ACTIVE_CREATION
        elif completion < 0.8:
            self.creative_phase = CreativePhase.REFLECTION
        else:
            self.creative_phase = CreativePhase.STYLE_DEVELOPMENT
            
    async def autonomous_create(self, iterations=100):
        """Main autonomous creation loop"""
        print("🚀 STARTING AUTONOMOUS AI ARTIST SESSION")
        print(f"Session ID: {self.session_id}")
        
        for i in range(iterations):
            print(f"\n--- Iteration {i+1}/{iterations} ---")
            
            # 1. Perceive
            perception = self.perceive()
            
            # 2. Update phase
            self.update_phase(perception)
            
            # 3. Plan action
            action = self.plan_action(perception)
            
            # 4. Execute and assess
            quality = self.execute_action(action)
            
            # 5. Learn from experience
            self.memory.add_experience(action, quality)
            
            # 6. Display progress
            print(f"🧠 Phase: {self.creative_phase.value}")
            print(f"🎨 Action: {action.tool} at ({action.x},{action.y}) - Quality: {quality:.2f}")
            print(f"📊 Canvas: {perception['canvas_stats']['completion']:.1%} complete, {perception['canvas_stats']['colors_used']} colors")
            print(f"🎯 Memory: {perception['memory_state']['experiences']} experiences, preferred tool: {perception['memory_state']['preferred_tool']}")
            
            # Display canvas every few iterations
            if (i + 1) % 5 == 0 or i == iterations - 1:
                self.canvas.display()
                
            # Brief pause for readability
            await asyncio.sleep(0.3)
            
        # Final session summary
        await self.session_summary()
        
    async def session_summary(self):
        """Display final session summary"""
        perception = self.perceive()
        
        print("\n" + "🌟"*20)
        print("AUTONOMOUS CREATION SESSION COMPLETE")
        print("🌟"*20)
        
        print(f"\n📈 Session Statistics:")
        print(f"   Canvas Completion: {perception['canvas_stats']['completion']:.1%}")
        print(f"   Pixels Painted: {perception['canvas_stats']['pixels_painted']}")
        print(f"   Colors Used: {perception['canvas_stats']['colors_used']}")
        print(f"   Creative Experiences: {perception['memory_state']['experiences']}")
        
        print(f"\n🧠 AI Learning Progress:")
        print(f"   Tools Mastered: {perception['memory_state']['tools_learned']}")
        print(f"   Preferred Tool: {perception['memory_state']['preferred_tool']}")
        print(f"   Tool Mastery Levels:")
        for tool, mastery in perception['memory_state']['tool_mastery'].items():
            print(f"      {tool}: {mastery}")
            
        print(f"\n🎨 Final Artistic Phase: {self.creative_phase.value}")
        
        # Save session
        self.save_session()
        print(f"\n💾 Session saved to: .zw/sessions/{self.session_id}.json")
        
    def save_session(self):
        """Save session data with proper type conversion"""

        def convert_types(obj):
            """Convert complex types to JSON-serializable Python types"""
            if isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        session_data = {
            'session_id': self.session_id,
            'canvas': self.canvas.canvas,
            'final_stats': convert_types(self.canvas.get_stats()),
            'memory_summary': convert_types(self.memory.get_memory_summary()),
            'final_phase': self.creative_phase.value,
            'experiences': convert_types(self.memory.short_term)
        }
        
        session_dir = Path(".zw/sessions")
        session_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = session_dir / f"{self.session_id}.json"
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Session saved successfully to: {session_file}")

async def main():
    """Main entry point"""
    print("🎨 TRIXEL COMPOSER: AUTONOMOUS AI ARTIST")
    print("=" * 50)
    print("🚀 Initializing autonomous creative intelligence...")
    
    composer = TerminalTrixelComposer()
    
    # Show initial empty canvas
    composer.canvas.display()
    
    print("\n⏳ Starting autonomous creation in 3 seconds...")
    await asyncio.sleep(3)
    
    # Begin autonomous creation
    await composer.autonomous_create(30)
    
    print("\n✨ Autonomous AI artist session complete!")
    print("🎯 Check .zw/sessions/ folder for saved session data")

# PNG Export Patch - Add this before the main section
def apply_png_patch():
    """Apply PNG export capabilities to existing classes"""
    
    def to_png(self, output_dir=".", session_id="", scale=20):
        try:
            from PIL import Image
        except ImportError:
            print("⚠️ Pillow not available; skipping PNG export.")
            return None

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            scaled_rows = []
            for row in self.canvas:
                expanded_row = []
                for pixel in row:
                    expanded_row.extend([tuple(pixel)] * scale)
                for _ in range(scale):
                    scaled_rows.extend(expanded_row)

            img = Image.new('RGB', (self.width * scale, self.height * scale))
            img.putdata(scaled_rows)

            filename = f"trixel_{session_id}_{int(time.time())}.png"
            png_file = output_path / filename

            img.save(png_file)
            print(f"🎨 Artwork exported: {png_file}")
            return png_file
        except Exception as e:
            print(f"❌ PNG export failed: {e}")
            return None
    
    # Add PNG export to TerminalCanvas
    TerminalCanvas.to_png = to_png
    
    # Enhanced save_session with PNG export
    original_save = TerminalTrixelComposer.save_session
    
    def enhanced_save_with_png(self):
        # Call original save method
        original_save(self)
        
        # Export PNG artwork
        artwork_dir = Path(".zw/artwork")
        artwork_dir.mkdir(parents=True, exist_ok=True)
        
        png_file = self.canvas.to_png(
            output_dir=str(artwork_dir),
            session_id=self.session_id
        )
        
        if png_file:
            print(f"🎨✨ Your AI's artwork is now visible: {png_file}")
    
    TerminalTrixelComposer.save_session = enhanced_save_with_png

# Apply the PNG patch
apply_png_patch()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Session interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you have Python 3.7+ available; optional Pillow enables PNG export")


# Empire Art Bridge: AI-to-AI Artistic Collaboration System
# Optional Empire integration stubs for offline compatibility.
# Requires aiohttp and remote services; disabled by default.

def enable_empire_mode():
    print('🏰 Empire mode is unavailable because optional dependencies are not installed.')
    print('Run `pip install aiohttp` and connect to an Empire broker to enable it.')


async def launch_empire_collaborative_session():
    raise RuntimeError('Empire collaborative mode requires optional aiohttp dependency')
