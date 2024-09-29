//
//  FitMLApp.swift
//  FitML
//
//  Created by Yahya Naveed Saleem on 29.09.24.
//
import SwiftUI

@main
struct FitMLApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
