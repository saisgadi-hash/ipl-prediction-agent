import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // Brand colours
  static const Color bg = Color(0xFF0A0A1A);
  static const Color surface = Color(0xFF12122A);
  static const Color card = Color(0xFF1A1A3E);
  static const Color accent = Color(0xFF6366F1);   // Indigo
  static const Color accentPink = Color(0xFFEC4899);
  static const Color textPrimary = Color(0xFFE2E8F0);
  static const Color textSecondary = Color(0xFF94A3B8);
  static const Color success = Color(0xFF22C55E);
  static const Color warning = Color(0xFFF59E0B);
  static const Color error = Color(0xFFEF4444);

  static ThemeData get darkTheme => ThemeData(
    brightness: Brightness.dark,
    scaffoldBackgroundColor: bg,
    colorScheme: const ColorScheme.dark(
      primary: accent,
      secondary: accentPink,
      surface: surface,
    ),
    textTheme: GoogleFonts.interTextTheme(ThemeData.dark().textTheme).copyWith(
      headlineLarge: GoogleFonts.inter(fontSize: 28, fontWeight: FontWeight.w700, color: textPrimary),
      headlineMedium: GoogleFonts.inter(fontSize: 22, fontWeight: FontWeight.w600, color: textPrimary),
      titleLarge: GoogleFonts.inter(fontSize: 18, fontWeight: FontWeight.w600, color: textPrimary),
      titleMedium: GoogleFonts.inter(fontSize: 16, fontWeight: FontWeight.w500, color: textPrimary),
      bodyLarge: GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.w400, color: textPrimary),
      bodyMedium: GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w400, color: textSecondary),
      bodySmall: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w400, color: textSecondary),
    ),
    cardTheme: CardTheme(
      color: card,
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
    navigationBarTheme: NavigationBarThemeData(
      labelTextStyle: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: accent);
        }
        return GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w400, color: textSecondary);
      }),
      iconTheme: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return const IconThemeData(color: accent, size: 24);
        }
        return const IconThemeData(color: textSecondary, size: 24);
      }),
    ),
  );

  // Team colours
  static const Map<String, Color> teamColors = {
    'Chennai Super Kings': Color(0xFFFFD700),
    'Mumbai Indians': Color(0xFF004BA0),
    'Royal Challengers Bengaluru': Color(0xFFEC1C24),
    'Kolkata Knight Riders': Color(0xFF3A225D),
    'Delhi Capitals': Color(0xFF004C93),
    'Punjab Kings': Color(0xFFED1B24),
    'Rajasthan Royals': Color(0xFFEA1A85),
    'Sunrisers Hyderabad': Color(0xFFF26522),
    'Lucknow Super Giants': Color(0xFF00AEEF),
    'Gujarat Titans': Color(0xFF1B2133),
  };

  static Color getTeamColor(String team) => teamColors[team] ?? accent;

  // Gradient for accent buttons and cards
  static const LinearGradient accentGradient = LinearGradient(
    colors: [accent, accentPink],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
}
