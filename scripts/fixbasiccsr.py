import os
import site

# Find the basicsr degradations.py file
for path in site.getsitepackages():
    target = os.path.join(path, "basicsr", "data", "degradations.py")
    if os.path.exists(target):
        print(f"Found: {target}")
        with open(target, "r") as f:
            content = f.read()

        old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
        new = "from torchvision.transforms.functional import rgb_to_grayscale"

        if old in content:
            content = content.replace(old, new)
            with open(target, "w") as f:
                f.write(content)
            print("Fixed! Line updated successfully.")
        elif new in content:
            print("Already fixed — no changes needed.")
        else:
            print("Line not found — may need manual inspection.")
        break
else:
    print("basicsr not found in site-packages. Check your venv.")