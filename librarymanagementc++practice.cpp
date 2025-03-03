#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;

class Book {
public:
    int id;
    string title;
    string author;
    bool issued;

    Book(int bookID, string bookTitle, string bookAuthor) {
        id = bookID;
        title = bookTitle;
        author = bookAuthor;
        issued = false;
    }
};

class Library {
private:
    vector<Book> books;
    string filename = "library_data.txt";

public:
    Library() {
        loadFromFile();
    }

    void addBook(int id, string title, string author) {
        books.push_back(Book(id, title, author));
        saveToFile();
    }

    void issueBook(int id) {
        for (auto &book : books) {
            if (book.id == id && !book.issued) {
                book.issued = true;
                saveToFile();
                cout << "Book issued successfully!" << endl;
                return;
            }
        }
        cout << "Book not available or already issued." << endl;
    }

    void returnBook(int id) {
        for (auto &book : books) {
            if (book.id == id && book.issued) {
                book.issued = false;
                saveToFile();
                cout << "Book returned successfully!" << endl;
                return;
            }
        }
        cout << "Invalid book ID or book not issued." << endl;
    }

    void displayBooks() {
        cout << "\nLibrary Books:\n";
        for (const auto &book : books) {
            cout << "ID: " << book.id << " | Title: " << book.title << " | Author: " << book.author << " | Issued: " << (book.issued ? "Yes" : "No") << endl;
        }
    }

    void searchBook(string keyword) {
        cout << "\nSearch Results:" << endl;
        for (const auto &book : books) {
            if (book.title.find(keyword) != string::npos || book.author.find(keyword) != string::npos) {
                cout << "ID: " << book.id << " | Title: " << book.title << " | Author: " << book.author << endl;
            }
        }
    }

    void saveToFile() {
        ofstream file(filename);
        for (const auto &book : books) {
            file << book.id << "|" << book.title << "|" << book.author << "|" << book.issued << "\n";
        }
        file.close();
    }

    void loadFromFile() {
        ifstream file(filename);
        if (!file) return;
        books.clear();
        int id;
        string title, author;
        bool issued;
        while (file >> id) {
            file.ignore();
            getline(file, title, '|');
            getline(file, author, '|');
            file >> issued;
            books.push_back(Book(id, title, author));
            books.back().issued = issued;
        }
        file.close();
    }
};

int main() {
    Library lib;
    int choice, id;
    string title, author, keyword;
    
    while (true) {
        cout << "\nLibrary Management System";
        cout << "\n1. Add Book";
        cout << "\n2. Issue Book";
        cout << "\n3. Return Book";
        cout << "\n4. Display Books";
        cout << "\n5. Search Book";
        cout << "\n6. Exit";
        cout << "\nEnter choice: ";
        cin >> choice;
        cin.ignore();
        
        switch (choice) {
            case 1:
                cout << "Enter Book ID: ";
                cin >> id;
                cin.ignore();
                cout << "Enter Title: ";
                getline(cin, title);
                cout << "Enter Author: ";
                getline(cin, author);
                lib.addBook(id, title, author);
                break;
            case 2:
                cout << "Enter Book ID to issue: ";
                cin >> id;
                lib.issueBook(id);
                break;
            case 3:
                cout << "Enter Book ID to return: ";
                cin >> id;
                lib.returnBook(id);
                break;
            case 4:
                lib.displayBooks();
                break;
            case 5:
                cout << "Enter keyword to search: ";
                cin.ignore();
                getline(cin, keyword);
                lib.searchBook(keyword);
                break;
            case 6:
                cout << "Exiting program..." << endl;
                return 0;
            default:
                cout << "Invalid choice! Try again." << endl;
        }
    }
}
