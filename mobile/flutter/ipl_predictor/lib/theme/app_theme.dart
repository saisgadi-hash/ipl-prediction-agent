import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // Brand colours — Clean & Colourful
  static const Color bg = Color(0xFFFAFBFE);
  static const Color surface = Color(0xFFF0F4FF);
  static const Color card = Color(0xFFFFFFFF);
  static const Color accent = Color(0xFF6C5CE7);       // Vivid purple
  static const Color accentLight = Color(0xFFA29BFE);   // Soft purple
  static const Color mint = Color(0xFF00B894);           // Mint green
  static const Color golden = Color(0xFFFDCB6E);         // Golden yellow
  static const Color coral = Color(0xFFE17055);           // Soft coral
  static const Color skyBlue = Color(0xFF74B9FF);         // Sky blue
  static const Color textPrimary = Color(0xFF2D3748);
  static const Color textSecondary = Color(0xFF718096);
  static const Color border = Color(0xFFEDF2F7);
  static const Color success = Color(0xFF00B894);
  static const Color warning = Color(0xFFFDCB6E);
  static const Color error = Color(0xFFE17055);

  static ThemeData get lightTheme => ThemeData(
    brightness: Brightness.light,
    scaffoldBackgroundColor: bg,
    colorScheme: const ColorScheme.light(
      primary: accent,
      secondary: mint,
      surface: card,
    ),
    textTheme: GoogleFonts.interTextTheme(ThemeData.light().textTheme).copyWith(
      headlineLarge: GoogleFonts.inter(fontSize: 28, fontWeight: FontWeight.w700, color: textPrimary),
      headlineMedium: GoogleFonts.inter(fontSize: 22, fontWeight: FontWeight.w600, color: textPrimary),
      titleLarge: GoogleFonts.inter(fontSize: 18, fontWeight: FontWeight.w600, color: textPrimary),
      titleMedium: GoogleFonts.inter(fontSize: 16, fontWeight: FontWeight.w500, color: textPrimary),
      bodyLarge: GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.w400, color: textPrimary),
      bodyMedium: GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w400, color: textSecondary),
      bodySmall: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w400, color: textSecondary),
    ),
    cardTheme: CardThemeData(
      color: card,
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
    ),
    navigationBarTheme: NavigationBarThemeData(
      backgroundColor: Colors.white,
      surfaceTintColor: Colors.transparent,
      indicatorColor: accent.withOpacity(0.1),
      labelTextStyle: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w700, color: accent);
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

  // Vibrant team colours
  static const Map<String, Color> teamColors = {
    'Chennai Super Kings': Color(0xFFF5A623),
    'Mumbai Indians': Color(0xFF0984E3),
    'Royal Challengers Bengaluru': Color(0xFFE74C3C),
    'Kolkata Knight Riders': Color(0xFF8E44AD),
    'Delhi Capitals': Color(0xFF2980B9),
    'Punjab Kings': Color(0xFFE74C3C),
    'Rajasthan Royals': Color(0xFFE84393),
    'Sunrisers Hyderabad': Color(0xFFF39C12),
    'Lucknow Super Giants': Color(0xFF00CEC9),
    'Gujarat Titans': Color(0xFF636E72),
  };

  // Light tints for card backgrounds
  static const Map<String, Color> teamColorsLight = {
    'Chennai Super Kings': Color(0xFFFEF3E0),
    'Mumbai Indians': Color(0xFFE3F2FD),
    'Royal Challengers Bengaluru': Color(0xFFFFEBEE),
    'Kolkata Knight Riders': Color(0xFFF3E5F5),
    'Delhi Capitals': Color(0xFFE1F5FE),
    'Punjab Kings': Color(0xFFFFEBEE),
    'Rajasthan Royals': Color(0xFFFCE4EC),
    'Sunrisers Hyderabad': Color(0xFFFFF8E1),
    'Lucknow Super Giants': Color(0xFFE0F7FA),
    'Gujarat Titans': Color(0xFFECEFF1),
  };

  static Color getTeamColor(String team) => teamColors[team] ?? accent;
  static Color getTeamColorLight(String team) => teamColorsLight[team] ?? surface;

  // Gradient for accent buttons
  static const LinearGradient accentGradient = LinearGradient(
    colors: [accent, accentLight],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  // Colourful gradient for hero cards
  static const LinearGradient heroGradient = LinearGradient(
    colors: [Color(0xFF6C5CE7), Color(0xFF00B894), Color(0xFF74B9FF)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
}
