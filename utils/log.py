def print_model_info(models):
    """Print model architecture information."""
    print("\n" + "=" * 50)
    print("MODEL ARCHITECTURES")
    print("=" * 50)
    
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n{name} ({param_count:,} parameters):")
        print(model)
    
    print("=" * 50)