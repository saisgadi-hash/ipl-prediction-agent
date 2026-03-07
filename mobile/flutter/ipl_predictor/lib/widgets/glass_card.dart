import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

/// Reusable glass-morphism card matching the dashboard aesthetic.
class GlassCard extends StatelessWidget {
  final Widget child;
  final EdgeInsets padding;

  const GlassCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(16),
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: AppTheme.card.withOpacity(0.6),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF2A2A4A).withOpacity(0.5)),
      ),
      child: child,
    );
  }
}
