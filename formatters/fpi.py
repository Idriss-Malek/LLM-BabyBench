# formatters/fpi.py

import gymnasium as gym
from gymnasium import Env

from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Ball, Box

from formatters.base import EnvFormatter
from formatters.utils import agent_direction


class FPIFormatter(EnvFormatter):
    def __init__(self, style="structured", agent_view_size=7):
        """
        FPI (Full mission, Partial observability, Interactive) Formatter
        
        Args:
            style: Format style - "structured", "narrative", or "json"
            agent_view_size: Size of agent's view (default 7x7)
        """
        self.style = style
        self.agent_view_size = agent_view_size
    
    def format(self, env: Env) -> str:
        """Format environment with partial observability"""
        
        # Get basic environment info 
        agent_pos = tuple(int(x) for x in env.unwrapped.agent_pos)
        agent_front_pos = tuple(int(x) for x in env.unwrapped.front_pos)
        agent_dir = int(env.unwrapped.agent_dir)
        mission = str(env.unwrapped.mission)
        
        # Get ALL objects in environment with their absolute positions
        all_objects = []
        grid_width = env.unwrapped.grid.width
        for t, obj in enumerate(env.unwrapped.grid.grid):
            if isinstance(obj, (Door, Key, Ball, Box)):
                # Convert linear index to absolute 2D coordinates
                abs_x = t % grid_width
                abs_y = t // grid_width

                
                lock_status = f", locked={obj.is_locked}" if isinstance(obj, Door) else ""
                
                all_objects.append({
                    "obj_instance": obj,  # Keep reference to original object
                    "type": obj.type,
                    "color": obj.color,
                    "position": (abs_x, abs_y),
                    "lock_status": lock_status
                })

        #print(all_objects)
        
        # Filter to only visible objects using coordinate-based visibility check
        visible_objects = []
        for obj_data in all_objects:
            if self._is_object_visible(obj_data["position"], agent_pos, agent_dir):
                # Remove obj_instance before adding to visible_objects (not needed for formatting)
                visible_obj = {k: v for k, v in obj_data.items() if k != "obj_instance"}
                visible_objects.append(visible_obj)
        
        # Format based on style
        if self.style == "structured":
            return self._format_structured(agent_pos, agent_dir, agent_front_pos, mission, visible_objects)
        elif self.style == "narrative":
            return self._format_narrative(agent_pos, agent_dir, agent_front_pos, mission, visible_objects)
        elif self.style == "json":
            return self._format_json(agent_pos, agent_dir, agent_front_pos, mission, visible_objects)
        else:
            raise ValueError(f"Unknown style: {self.style}")

    def _is_object_visible(self, obj_pos, agent_pos, agent_dir):
        """Check if an object at obj_pos is visible from agent's position and direction.

        KNOWN LIMITATION: this is a pure view-frustum test and does NOT model
        wall occlusion. Objects geometrically inside the view cone but behind a
        wall (e.g. in an adjacent room) are still reported as visible. A correct
        fix requires line-of-sight / gen_obs_grid integration.
        """
        dx = obj_pos[0] - agent_pos[0] 
        dy = obj_pos[1] - agent_pos[1]

        # Calculate forward and right distances based on direction
        if agent_dir == 0:    # East
            forward = dx
            right = dy
        elif agent_dir == 1:  # South  
            forward = dy
            right = -dx
        elif agent_dir == 2:  # West
            forward = -dx
            right = -dy
        else:                 # North
            forward = -dy
            right = dx
        
        # Check if within view bounds
        max_forward = self.agent_view_size - 1  # 6 for 7x7
        max_side = self.agent_view_size // 2    # 3 for 7x7
        
        return (0 <= forward <= max_forward and -max_side <= right <= max_side)

    def _format_narrative(self, agent_pos, agent_dir, agent_front_pos, mission, visible_objects):
        """Format in narrative style"""

        context = f"You are at position {agent_pos}, facing {agent_direction(agent_dir)} toward {agent_front_pos}. "

        if visible_objects:
            context += "In your view, you can see: "
            obj_descriptions = []
            for obj in visible_objects:
                desc = f"a {obj['color']} {obj['type']} at {obj['position']}"
                if obj['lock_status']:
                    desc += f" ({obj['lock_status'].replace(', ', '').replace('=', ' is ')})"
                obj_descriptions.append(desc)
            context += ", ".join(obj_descriptions) + ". "
        else:
            context += "You cannot see any objects in your current view. "
            
        context += f"Your mission is: {mission}"
        return context

    def _format_structured(self, agent_pos, agent_dir, agent_front_pos, mission, visible_objects):
        """Format in structured style"""
        
        config = []
        config.append(f"- Your position: {agent_pos}")
        config.append(f"- Your direction: {agent_direction(agent_dir)} (toward {agent_front_pos})")
        
        # Visible objects
        if visible_objects:
            config.append("- Visible objects:")
            for obj in visible_objects:
                config.append(f"  - {obj['type']}, color={obj['color']}, position={obj['position']}{obj['lock_status']}")
        else:
            config.append("- Visible objects: none")
            
        config.append(f"- Mission: '{mission}'")
        
        return "\n".join(config) 
    
    def _format_json(self, agent_pos, agent_dir, agent_front_pos, mission, visible_objects):
        """Format in JSON style"""
        import json
        
        data = {
            "agent": {
                "position": agent_pos,
                "direction": agent_dir,
                "direction_name": agent_direction(agent_dir),
                "front_position": agent_front_pos
            },
            "visible_objects": [
                {
                    "type": obj["type"],
                    "color": obj["color"], 
                    "position": obj["position"],
                    "locked": "locked=True" in obj["lock_status"] if obj["lock_status"] else False
                }
                for obj in visible_objects
            ],
            "mission": mission
        }
        
        return json.dumps(data, indent=2)
    